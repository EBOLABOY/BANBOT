#!/usr/bin/env python
# 修复lstm_model.py文件中的缩进问题

def fix_indentation():
    with open('lstm_model.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 修复区域1：验证方向准确率计算部分（第1140-1165行左右）
    validation_dir_acc_fixed = [
        '                    # 计算总体方向准确率\n',
        '                    val_dir_acc = correct_dirs / total_dirs * 100 if total_dirs > 0 else 50.0\n',
        '                else:\n',
        '                    # 没有序列维度，视为批次数据\n',
        '                    val_dir_acc = 50.0  # 默认值\n',
        '            else:\n',
        '                # 2D数据 [batch, feature]\n',
        '                # 计算相邻样本间的差异\n',
        '                pred_diff = all_val_preds[1:] - all_val_preds[:-1]\n',
        '                target_diff = all_val_targets[1:] - all_val_targets[:-1]\n',
        '                \n',
        '                # 只考虑目标有明确变化方向的点\n',
        '                valid_idx = (target_diff != 0).flatten()\n',
        '                \n',
        '                if valid_idx.sum() > 0:\n',
        '                    # 找出方向预测正确的比例\n',
        '                    dir_match = (torch.sign(pred_diff) == torch.sign(target_diff)).flatten()\n',
        '                    val_dir_acc = dir_match[valid_idx].float().mean().item() * 100\n',
        '                else:\n',
        '                    val_dir_acc = 50.0\n',
        '        else:\n',
        '            val_dir_acc = 50.0  # 如果验证集太小，使用默认值\n',
    ]
    
    # 替换区域1
    if len(lines) >= 1165:
        lines[1140:1165] = validation_dir_acc_fixed
    
    # 修复区域2：数据增强部分（第2720-2733行左右）
    augmentation_fixed = [
        '    if disable_augmentation:\n',
        '        print("数据增强功能已禁用，使用原始数据进行训练")\n',
        '        X_train_aug, y_train_aug = X_train, y_train\n',
        '    elif mem_gb >= 8:\n',
        '        print(f"显存充足({mem_gb:.1f}GB)，启用数据增强...")\n',
        '        aug_factor = min(int(mem_gb / 4), 4)  # 根据显存大小决定增强倍数\n',
        '        print(f"计划增强训练集至 {aug_factor}x 原始大小")\n',
        '        X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train, num_augmentations=aug_factor)\n',
        '        print(f"增强后训练集大小: {X_train_aug.shape}")\n',
        '    else:\n',
        '        print(f"显存有限({mem_gb:.1f}GB)，使用原始数据进行训练")\n',
        '        X_train_aug, y_train_aug = X_train, y_train\n',
        '    \n',
        '    # 将数据转换为张量并创建数据加载器\n',
    ]
    
    # 替换区域2
    if len(lines) >= 2733:
        lines[2720:2733] = augmentation_fixed
    
    # 修复区域3：模型选择if-else语句（第2765-2768行左右）
    model_selection_fixed = [
        '    if model_type.lower() == \'advanced\':\n',
        '        model = model_class(input_size=input_size, seq_len=seq_len).to(device)\n',
        '    else:\n',
        '        model = model_class(input_size=input_size).to(device)\n',
        '    \n',
    ]
    
    # 替换区域3
    if len(lines) >= 2768:
        lines[2764:2769] = model_selection_fixed
    
    # 写回文件
    with open('lstm_model.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("已修复lstm_model.py中的缩进问题")

if __name__ == "__main__":
    fix_indentation() 