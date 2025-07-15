import os
import torch

# GPU 1번 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("=== GPU 설정 확인 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 테스트 텐서 생성
    device = torch.device("cuda:0")
    x = torch.randn(10, 10).to(device)
    print(f"Tensor device: {x.device}")
    print(f"실제 사용 중인 GPU: {torch.cuda.get_device_name(x.device)}")

print("\n=== nvidia-smi로 확인해보세요 ===")
print("터미널에서 'nvidia-smi' 명령어를 실행하여")
print("어떤 GPU에서 메모리가 사용되고 있는지 확인하세요.")