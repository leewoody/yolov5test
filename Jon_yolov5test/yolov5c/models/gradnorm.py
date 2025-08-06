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
        avg_grad_norm = torch.mean(grad_norms)
        
        # Update task weights
        new_weights = avg_grad_norm * relative_inverse_training_rate / grad_norms
        new_weights = torch.clamp(new_weights, min=0.1, max=10.0)
        
        # Normalize weights
        new_weights = new_weights / torch.mean(new_weights) * self.num_tasks
        
        self.task_weights = new_weights
        
        return new_weights

class GradNormLoss(nn.Module):
    """
    GradNorm Loss wrapper for YOLOv5WithClassification
    """
    
    def __init__(self, 
                 detection_loss_fn: nn.Module,
                 classification_loss_fn: nn.Module,
                 alpha: float = 1.5,
                 initial_detection_weight: float = 1.0,
                 initial_classification_weight: float = 0.01):
        super(GradNormLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.classification_loss_fn = classification_loss_fn
        
        # Initialize GradNorm
        self.gradnorm = GradNorm(
            num_tasks=2,
            alpha=alpha,
            initial_task_weights=[initial_detection_weight, initial_classification_weight]
        )
        
        # Loss history
        self.detection_loss_history = []
        self.classification_loss_history = []
        
    def forward(self, 
                detection_outputs: torch.Tensor,
                classification_outputs: torch.Tensor,
                detection_targets: torch.Tensor,
                classification_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with GradNorm loss balancing
        
        Args:
            detection_outputs: Detection model outputs
            classification_outputs: Classification model outputs
            detection_targets: Detection targets
            classification_targets: Classification targets
            
        Returns:
            Total loss and loss information
        """
        # Compute individual losses
        detection_loss_result = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Handle detection loss (might be tuple or tensor)
        if isinstance(detection_loss_result, tuple):
            detection_loss = detection_loss_result[0]  # Take first element if tuple
        else:
            detection_loss = detection_loss_result
        
        # Store loss history
        self.detection_loss_history.append(detection_loss.item())
        self.classification_loss_history.append(classification_loss.item())
        
        # Get current task weights
        current_losses = [detection_loss.item(), classification_loss.item()]
        
        # Update weights using GradNorm (every 10 steps)
        if len(self.detection_loss_history) % 10 == 0:
            # Compute gradients for weight update
            detection_grad = torch.autograd.grad(detection_loss, detection_outputs, retain_graph=True)[0]
            classification_grad = torch.autograd.grad(classification_loss, classification_outputs, retain_graph=True)[0]
            
            gradients = [detection_grad, classification_grad]
            self.gradnorm.update_task_weights(current_losses, gradients)
        
        # Apply current weights
        detection_weight = self.gradnorm.task_weights[0]
        classification_weight = self.gradnorm.task_weights[1]
        
        # Compute weighted total loss
        total_loss = detection_weight * detection_loss + classification_weight * classification_loss
        
        # Return loss and information
        loss_info = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': detection_weight.item(),
            'classification_weight': classification_weight.item(),
            'detection_grad_norm': torch.norm(detection_grad).item() if 'detection_grad' in locals() else 0.0,
            'classification_grad_norm': torch.norm(classification_grad).item() if 'classification_grad' in locals() else 0.0
        }
        
        return total_loss, loss_info

class AdaptiveGradNormLoss(nn.Module):
    """
    Adaptive GradNorm Loss with momentum and constraints
    """
    
    def __init__(self, 
                 detection_loss_fn: nn.Module,
                 classification_loss_fn: nn.Module,
                 alpha: float = 1.5,
                 momentum: float = 0.9,
                 min_weight: float = 0.001,
                 max_weight: float = 10.0):
        super(AdaptiveGradNormLoss, self).__init__()
        
        self.detection_loss_fn = detection_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize GradNorm
        self.gradnorm = GradNorm(
            num_tasks=2,
            alpha=alpha,
            initial_task_weights=[1.0, 0.01]
        )
        
        # Adaptive parameters
        self.detection_weight_momentum = 1.0
        self.classification_weight_momentum = 0.01
        
        # Loss history
        self.detection_loss_history = []
        self.classification_loss_history = []
        
    def compute_relative_loss_ratio(self, current_losses: List[float]) -> float:
        """
        Compute relative loss ratio for adaptive adjustment
        """
        if len(self.detection_loss_history) < 10:
            return 1.0
        
        # Compute recent average losses
        recent_detection_loss = np.mean(self.detection_loss_history[-10:])
        recent_classification_loss = np.mean(self.classification_loss_history[-10:])
        
        # Compute ratio
        ratio = recent_detection_loss / (recent_classification_loss + 1e-8)
        
        return ratio
        
    def forward(self, 
                detection_outputs: torch.Tensor,
                classification_outputs: torch.Tensor,
                detection_targets: torch.Tensor,
                classification_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive GradNorm loss balancing
        """
        # Compute individual losses
        detection_loss_result = self.detection_loss_fn(detection_outputs, detection_targets)
        classification_loss = self.classification_loss_fn(classification_outputs, classification_targets)
        
        # Handle detection loss (might be tuple or tensor)
        if isinstance(detection_loss_result, tuple):
            detection_loss = detection_loss_result[0]  # Take first element if tuple
        else:
            detection_loss = detection_loss_result
        
        # Store loss history
        self.detection_loss_history.append(detection_loss.item())
        self.classification_loss_history.append(classification_loss.item())
        
        # Get current task weights
        current_losses = [detection_loss.item(), classification_loss.item()]
        
        # Update weights using GradNorm (every 5 steps)
        if len(self.detection_loss_history) % 5 == 0:
            # Compute gradients for weight update
            detection_grad = torch.autograd.grad(detection_loss, detection_outputs, retain_graph=True)[0]
            classification_grad = torch.autograd.grad(classification_loss, classification_outputs, retain_graph=True)[0]
            
            gradients = [detection_grad, classification_grad]
            new_weights = self.gradnorm.update_task_weights(current_losses, gradients)
            
            # Apply momentum
            self.detection_weight_momentum = (
                self.momentum * self.detection_weight_momentum + 
                (1 - self.momentum) * new_weights[0].item()
            )
            self.classification_weight_momentum = (
                self.momentum * self.classification_weight_momentum + 
                (1 - self.momentum) * new_weights[1].item()
            )
            
            # Apply constraints
            self.detection_weight_momentum = np.clip(
                self.detection_weight_momentum, 
                self.min_weight, 
                self.max_weight
            )
            self.classification_weight_momentum = np.clip(
                self.classification_weight_momentum, 
                self.min_weight, 
                self.max_weight
            )
        
        # Apply current weights
        detection_weight = self.detection_weight_momentum
        classification_weight = self.classification_weight_momentum
        
        # Compute weighted total loss
        total_loss = detection_weight * detection_loss + classification_weight * classification_loss
        
        # Return loss and information
        loss_info = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'classification_loss': classification_loss.item(),
            'detection_weight': detection_weight,
            'classification_weight': classification_weight,
            'detection_grad_norm': torch.norm(detection_grad).item() if 'detection_grad' in locals() else 0.0,
            'classification_grad_norm': torch.norm(classification_grad).item() if 'classification_grad' in locals() else 0.0,
            'relative_loss_ratio': self.compute_relative_loss_ratio(current_losses)
        }
        
        return total_loss, loss_info 