#!/usr/bin/env python3
"""
最終訓練監控腳本
持續監控兩個完整 50 epoch 訓練的進度
"""

import time
import os
import re
from datetime import datetime

def get_log_tail(log_file, lines=5):
    """獲取日誌文件的最後幾行"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()[-lines:]
        return []
    except:
        return []

def parse_progress(log_lines):
    """解析訓練進度"""
    current_epoch = None
    progress_pct = None
    losses = {}
    
    for line in log_lines:
        line = line.strip()
        
        # 解析 epoch 進度
        if "Epoch" in line and "%|" in line:
            try:
                # 提取 epoch 信息
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = f"{epoch_match.group(1)}/{epoch_match.group(2)}"
                
                # 提取進度百分比
                progress_match = re.search(r'(\d+)%\|', line)
                if progress_match:
                    progress_pct = int(progress_match.group(1))
                
                # 提取損失值
                if "Total=" in line:
                    total_match = re.search(r'Total=([\d.]+)', line)
                    if total_match:
                        losses['total'] = float(total_match.group(1))
                
                if "Det=" in line:
                    det_match = re.search(r'Det=([\d.]+)', line)
                    if det_match:
                        losses['detection'] = float(det_match.group(1))
                
                if "Cls=" in line:
                    cls_match = re.search(r'Cls=([\d.]+)', line)
                    if cls_match:
                        losses['classification'] = float(cls_match.group(1))
                
            except:
                pass
    
    return current_epoch, progress_pct, losses

def monitor_training():
    """監控兩個訓練"""
    print("\n" + "="*80)
    print(f"🔍 訓練監控報告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 監控修復版 GradNorm
    gradnorm_logs = get_log_tail("fixed_gradnorm_final_50.log", 10)
    gradnorm_epoch, gradnorm_progress, gradnorm_losses = parse_progress(gradnorm_logs)
    
    print(f"\n🎯 修復版 GradNorm 訓練:")
    if gradnorm_epoch:
        print(f"   📊 進度: {gradnorm_epoch} ({gradnorm_progress}%)")
        if gradnorm_losses:
            print(f"   📉 總損失: {gradnorm_losses.get('total', 'N/A'):.4f}")
            print(f"   🔍 檢測損失: {gradnorm_losses.get('detection', 'N/A'):.4f}")
            print(f"   🎯 分類損失: {gradnorm_losses.get('classification', 'N/A'):.4f}")
    else:
        print("   📋 狀態: 初始化或數據載入中...")
    
    # 監控雙獨立骨幹網路
    dual_logs = get_log_tail("dual_backbone_final_50.log", 10)
    dual_epoch, dual_progress, dual_losses = parse_progress(dual_logs)
    
    print(f"\n🏆 雙獨立骨幹網路訓練:")
    if dual_epoch:
        print(f"   📊 進度: {dual_epoch} ({dual_progress}%)")
        if dual_losses:
            print(f"   📉 總損失: {dual_losses.get('total', 'N/A'):.4f}")
            print(f"   🔍 檢測損失: {dual_losses.get('detection', 'N/A'):.4f}")
            print(f"   🎯 分類損失: {dual_losses.get('classification', 'N/A'):.4f}")
    else:
        print("   📋 狀態: 初始化或數據載入中...")
    
    # 性能對比
    if gradnorm_losses and dual_losses:
        print(f"\n📈 實時性能對比:")
        
        gradnorm_total = gradnorm_losses.get('total', 0)
        dual_total = dual_losses.get('total', 0)
        
        if gradnorm_total > 0 and dual_total > 0:
            if dual_total < gradnorm_total:
                improvement = ((gradnorm_total - dual_total) / gradnorm_total) * 100
                print(f"   🥇 雙獨立骨幹網路領先: {improvement:.1f}% 更低的總損失")
            else:
                improvement = ((dual_total - gradnorm_total) / dual_total) * 100
                print(f"   🥇 GradNorm 領先: {improvement:.1f}% 更低的總損失")
        
        # 檢測損失對比
        gradnorm_det = gradnorm_losses.get('detection', 0)
        dual_det = dual_losses.get('detection', 0)
        
        if gradnorm_det > 0 and dual_det > 0:
            det_diff = ((gradnorm_det - dual_det) / gradnorm_det) * 100
            if det_diff > 0:
                print(f"   🎯 檢測任務: 雙獨立骨幹網路檢測損失低 {det_diff:.1f}%")
            else:
                print(f"   🎯 檢測任務: GradNorm 檢測損失低 {-det_diff:.1f}%")
    
    print("\n" + "="*80)

def main():
    """主函數"""
    print("🚀 開始監控兩個完整 50 epoch 訓練...")
    print("📋 每分鐘更新一次進度")
    print("💡 按 Ctrl+C 停止監控")
    
    try:
        while True:
            monitor_training()
            time.sleep(60)  # 每分鐘檢查一次
    except KeyboardInterrupt:
        print("\n⏹️  監控已停止")
        print("✅ 兩個訓練將繼續在背景運行")

if __name__ == "__main__":
    main() 