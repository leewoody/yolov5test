import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

class GradNorm:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    Reference: Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    """
    
    def __init__(self, num_tasks: int, alpha: float = 1.5, initial_task_weights: List[float] = None):
        """
        Initialize GradNorm
        
        Args:
            num_tasks: Number of tasks (detection + classification)
            alpha: GradNorm parameter for controlling the rate of weight adjustment
            initial_task_weights: Initial weights for each task
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        # Initialize task weights
        if initial_task_weights is None:
            self.task_weights = torch.ones(num_tasks, requires_grad=False)
        else:
            self.task_weights = torch.tensor(initial_task_weights, requires_grad=False)
        
        # Initialize loss history
        self.initial_losses = None
        self.loss_history = []
        
        # Initialize average gradient norm
        self.avg_grad_norm = None
        
    def compute_relative_inverse_training_rate(self, current_losses: List[float]) -> torch.Tensor:
        """
        Compute relative inverse training rate for each task
        
        Args:
            current_losses: Current loss values for each task
            
        Returns:
            Relative inverse training rates
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(current_losses, requires_grad=False)
            return torch.ones(self.num_tasks)
        
        # Compute training rate for each task
        training_rates = torch.tensor(current_losses) / self.initial_losses
        
        # Compute average training rate
        avg_training_rate = torch.mean(training_rates)
        
        # Compute relative inverse training rate
        relative_inverse_training_rate = (training_rates / avg_training_rate) ** self.alpha
        
        return relative_inverse_training_rate
    
    def compute_grad_norms(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute gradient norms for each task
        
        Args:
            gradients: Gradients for each task
            
        Returns:
            Gradient norms for each task
        """
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                grad_norm = torch.norm(grad, p=2)
                grad_norms.append(grad_norm)
            else:
                grad_norms.append(torch.tensor(0.0))
        
        return torch.stack(grad_norms)
    
    def update_task_weights(self, 
                           current_losses: List[float], 
                           gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Update task weights using GradNorm
        
        Args:
            current_losses: Current loss values for each task
            gradients: Gradients for each task
            
        Returns:
            Updated task weights
        """
        # Compute relative inverse training rates
        relative_inverse_training_rate = self.compute_relative_inverse_training_rate(current_losses)
        
        # Compute gradient norms
        grad_norms = self.compute_grad_norms(gradients)
        
        # Compute average gradient norm
        if self.avg_grad_norm is None:
            self.avg_grad_norm = torch.mean(grad_norms)
        
        # Compute target gradient norms
        target_grad_norms = self.avg_grad_norm * relative_inverse_training_rate
        
        # Compute weight adjustment factors
        weight_adjustment = target_grad_norms / (grad_norms + 1e-8)
        
        # Update task weights
        new_weights = self.task_weights * weight_adjustment
        
        # Normalize weights to maintain average
        new_weights = new_weights / torch.mean(new_weights) * self.num_tasks
        
        # Update stored weights
        self.task_weights = new_weights.detach()
        
        return self.task_weights

class GradNormLoss(nn.Module):
    """
    GradNorm Loss wrapper for joint detection and classification
    """
    
    def __init__(self, 
                 detection_loss_fn: nn.Module,
                 classification_loss_fn: nn.Module,
                 alpha: float = 1.5,
                 initial_detection_weight: float = 1.0,
                 initial_classification_weight: float = 0.01):
        """
        Initialize GradNorm Loss
        
        Args:
            detection_loss_fn: Detection loss function
            classification_loss_fn: Classification loss function
            alpha: GradNorm parameter
            initial_detection_weight: Initial weight for detection task
            initial_classification_weight: Initial weight for classification task
        """
        super(GradNormLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.classification_loss_fn = classification_loss_fn
        
        # Initialize GradNorm
        initial_weights = [initial_detection_weight, initial_classification_weight]
        self.gradnorm = GradNorm(num_tasks=2, alpha=alpha, initial_task_weights=initial_weights)
        
        # Loss history
        self.detection_losses = []
        self.classification_losses = []
        
    def forward(self, 
                detection_outputs: torch.Tensor,
                classification_outputs: torch.Tensor,
                detection_targets: torch.Tensor,
                classification_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with GradNorm
        
        Args:
            detection_outputs: Detection model outputs
            classification_outputs: Classification model outputs
            detection_targets: Detection targets
            classification_targets: Classification targets
            
        Returns:
            Total loss and loss information
        """
        # Compute individual losses
        detection_loss = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Store losses for GradNorm
        self.detection_losses.append(detection_loss.item())
        self.classification_losses.append(classification_loss.item())
        
        # Get current task weights from GradNorm
        current_losses = [detection_loss.item(), classification_loss.item()]
        
        # For the first few iterations, use initial weights
        if len(self.detection_losses) < 10:
            weights = torch.tensor([1.0, 0.01])
        else:
            # Compute gradients for weight update
            detection_grad = torch.autograd.grad(detection_loss, detection_outputs, retain_graph=True)[0]
            classification_grad = torch.autograd.grad(classification_loss, classification_outputs, retain_graph=True)[0]
            
            gradients = [detection_grad, classification_grad]
            weights = self.gradnorm.update_task_weights(current_losses, gradients)
        
        # Apply weights to losses
        weighted_detection_loss = weights[0] * detection_loss
        weighted_classification_loss = weights[1] * classification_loss
        
        # Total loss
        total_loss = weighted_detection_loss + weighted_classification_loss
        
        # Return loss information
        loss_info = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': weights[0].item(),
            'classification_weight': weights[1].item(),
            'weighted_detection_loss': weighted_detection_loss.item(),
            'weighted_classification_loss': weighted_classification_loss.item()
        }
        
        return total_loss, loss_info

class AdaptiveGradNormLoss(nn.Module):
    """
    Enhanced GradNorm Loss with adaptive learning rate and momentum
    """
    
    def __init__(self, 
                 detection_loss_fn: nn.Module,
                 classification_loss_fn: nn.Module,
                 alpha: float = 1.5,
                 momentum: float = 0.9,
                 min_weight: float = 0.001,
                 max_weight: float = 10.0):
        """
        Initialize Adaptive GradNorm Loss
        
        Args:
            detection_loss_fn: Detection loss function
            classification_loss_fn: Classification loss function
            alpha: GradNorm parameter
            momentum: Momentum for weight updates
            min_weight: Minimum task weight
            max_weight: Maximum task weight
        """
        super(AdaptiveGradNormLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.alpha = alpha
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize weights
        self.detection_weight = nn.Parameter(torch.tensor(1.0))
        self.classification_weight = nn.Parameter(torch.tensor(0.01))
        
        # Loss history
        self.detection_losses = []
        self.classification_losses = []
        self.initial_losses = None
        
        # Momentum buffers
        self.detection_weight_momentum = 0.0
        self.classification_weight_momentum = 0.0
        
    def compute_relative_loss_ratio(self, current_losses: List[float]) -> float:
        """
        Compute relative loss ratio for adaptive weight adjustment
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(current_losses)
            return 1.0
        
        # Compute loss ratios
        loss_ratios = torch.tensor(current_losses) / self.initial_losses
        
        # Compute relative ratio
        relative_ratio = (loss_ratios[1] / loss_ratios[0]) ** self.alpha
        
        return relative_ratio.item()
    
    def forward(self, 
                detection_outputs: torch.Tensor,
                classification_outputs: torch.Tensor,
                detection_targets: torch.Tensor,
                classification_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive GradNorm
        """
        # Compute individual losses
        detection_loss = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Store losses
        self.detection_losses.append(detection_loss.item())
        self.classification_losses.append(classification_loss.item())
        
        # Adaptive weight adjustment
        if len(self.detection_losses) >= 10:
            current_losses = [detection_loss.item(), classification_loss.item()]
            relative_ratio = self.compute_relative_loss_ratio(current_losses)
            
            # Update classification weight based on relative performance
            target_weight = torch.clamp(
                self.detection_weight * relative_ratio,
                min=self.min_weight,
                max=self.max_weight
            )
            
            # Apply momentum
            self.classification_weight_momentum = (
                self.momentum * self.classification_weight_momentum + 
                (1 - self.momentum) * (target_weight - self.classification_weight)
            )
            
            # Update weight
            self.classification_weight.data = torch.clamp(
                self.classification_weight + self.classification_weight_momentum,
                min=self.min_weight,
                max=self.max_weight
            )
        
        # Apply weights
        weighted_detection_loss = self.detection_weight * detection_loss
        weighted_classification_loss = self.classification_weight * classification_loss
        
        # Total loss
        total_loss = weighted_detection_loss + weighted_classification_loss
        
        # Return loss information
        loss_info = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': self.detection_weight.item(),
            'classification_weight': self.classification_weight.item(),
            'weighted_detection_loss': weighted_detection_loss.item(),
            'weighted_classification_loss': weighted_classification_loss.item()
        }
        
        return total_loss, loss_info 