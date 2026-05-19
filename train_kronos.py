from data.dataloader import event_set
from data.sliding_window_dataloader import sliding_window_set
import os
from model.CAMEF4P19K import CAMEF, train_three_stage
# from model.GPT4MTS_50 import CAMEF, train, test
import torch
import warnings
from model.kronos import Kronos, KronosTokenizer, KronosAdapter  # ensure import path matches your repo

# 屏蔽所有警告
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

torch.autograd.set_detect_anomaly(True)

tokenizer = KronosTokenizer.from_pretrained("/home/yang/Research/Kronos/Kronos-Tokenizer-base/",local_files_only=True)
kronos = Kronos.from_pretrained("/home/yang/Research/Kronos/Kronos-base",local_files_only=True)

# 2) Build adapter
kronos_adapter = KronosAdapter(kronos, tokenizer, device="cuda:0", max_context=512, clip=5)


if __name__ == "__main__":

    seq_len=35
    pred_len=140
    event_id  = 0
    series_id = 'SP500'
    # moment_model = os.path.join('model','moment')
    # bert_model = os.path.join('model','longformer')
    # gpt_model = os.path.join('model','gpt2')
    # gpt_model = "openai-community/gpt2"
    # bert_model = "allenai/longformer-base-4096"
    # moment_model = "AutonLab/MOMENT-1-large"
    # moment_model = "/home/yang/Research/CAMEF/baselines/moment/moment-epoch3-sp500-len35/"

    batch_size = 32
    num_epochs = 2
    stage1_epochs = 2
    stage2_epochs = 2
    window=500
    stride=400
    event_set = event_set(
        seq_len, pred_len,


        event_id=event_id,
        series_id= series_id,

        shuffle = False,
        batch_size = batch_size,

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
    # llama_name = "/home/yang/Research/CAMEF/baselines/Qwen2.5-3B-Instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Phi-4-mini-instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Falcon3-3B-Instruct/"
    # llama_name = "/home/yang/Research/CAMEF/baselines/Phi-3.5-mini-instruct/"

    # Replace ONLY this block ↓
    model = CAMEF(
        llama_name=llama_name,
        # moment=moment_model,                    # remove or keep but unused when ts_backbone="kronos"
        kronos_model="/home/yang/Research/Kronos/Kronos-base",  # NEW: a small switch
        kronos_tokenizer="/home/yang/Research/Kronos/Kronos-Tokenizer-base",  # NEW: pass the adapter
        seq_len=seq_len,
        pred_len=pred_len,
        d=event_set.d,
        window=window,
        stride=stride,
        batch_size=batch_size,
        decoder_layers=3,
        decoder_heads=8,
        use_ts_memory=False,
        max_token_num=1024,
        # use_ts_memory=True,
        # modality_mode = "text"
    )

    output_path = "/home/yang/Research/CAMEF/model_output/CAMEF+_llama3_kronos_camef4p15k_crosshead8_sp500_len35/"

    train_three_stage(model, train_loader, test_loader, vali_loader,
                      sw_train_loader=sw_data.train_loader,  # Sliding window for Stage 1
                      sw_vali_loader=sw_data.vali_loader,  # Sliding window for Stage 1
                      stage1_epochs=stage1_epochs,
                      stage2_epochs=stage2_epochs,
                      num_epochs=num_epochs,
                      output_path=output_path,
                      contrastive_lambda=1.0,
                      eval_every_steps=None)
