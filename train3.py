from data.dataloader import event_set
from data.sliding_window_dataloader import sliding_window_set
import os
from model.CAMEF4P19L import CAMEF, train_three_stage
# from model.GPT4MTS_50 import CAMEF, train, test
import torch
import warnings

# 屏蔽所有警告
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.utils._pytree._register_pytree_node is deprecated")

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    seq_len = 35
    pred_len = 140
    event_id = 0
    series_id = 'SP500'
    # moment_model = os.path.join('model','moment')
    # bert_model = os.path.join('model','longformer')
    # gpt_model = os.path.join('model','gpt2')
    # gpt_model = "openai-community/gpt2"
    # bert_model = "allenai/longformer-base-4096"
    moment_model = "/home/yang/Research/CAMEF/baselines/moment/MOMETN-1-large/"
    # moment_model = "/home/yang/Research/CAMEF/baselines/moment/moment-epoch3-sp500-len35/"

    batch_size = 32
    num_epochs = 2
    stage1_epochs = 2
    stage2_epochs = 2
    window = 500
    stride = 400
    event_set = event_set(
        seq_len, pred_len,

        event_id=event_id,
        series_id=series_id,

        shuffle=False,
        batch_size=batch_size,

        scale=True,
        event_dir='data/event',
        series_dir='data/series',

    )

    sw_data = sliding_window_set(
        seq_len=seq_len, pred_len=pred_len,
        series_id=series_id, series_dir='data/series',
        stride=1, batch_size=batch_size
    )

    train_loader, test_loader, vali_loader = event_set.train_loader, event_set.test_loader, event_set.vali_loader
    # Add a llama model id/path (new)
    llama_name = "/home/yang/Research/CAMEF/llama3-2-3B/"  # or your local path
    # llama_name = "/home/yang/Research/CAMEF/Llama-3.1-8B/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Qwen2.5-3B-Instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Phi-4-mini-instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Phi-3.5-mini-instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Falcon3-3B-Instruct/"

    # Replace ONLY this block ↓
    model = CAMEF(
        llama_name=llama_name,  # NEW: use Llama 3.1 8B as the text encoder
        moment=moment_model,  # same MOMENT path
        seq_len=seq_len,
        pred_len=pred_len,
        d=event_set.d,
        window=window,  # token window for text chunks
        stride=stride,  # token stride for text chunks
        batch_size=batch_size,
        decoder_layers=3,  # (optional) tiny Transformer decoder depth
        decoder_heads=8,  # (optional) heads; keep 8 or 16 for 1024 dim
        use_ts_memory=False,
        max_token_num=1024,
    )

    output_path = "/home/yang/Research/CAMEF/model_output/final/CAMEF+_llama3_moment_camef4p19l_nasdaq_f140/"

    train_three_stage(model, train_loader, test_loader, vali_loader,
                      sw_train_loader = sw_data.train_loader,  # Sliding window for Stage 1
                      sw_vali_loader = sw_data.vali_loader,  # Sliding window for Stage 1
                      stage1_epochs = stage1_epochs,
                      stage2_epochs = stage2_epochs,
                      num_epochs = num_epochs,
                      output_path = output_path,
                      contrastive_lambda = 1.0,
                      eval_every_steps = None)
