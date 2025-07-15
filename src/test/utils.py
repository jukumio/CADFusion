import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model

IGNORE_INDEX = -100
MAX_LENGTH = 512
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer), mean_resizing=False)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def prepare_model_and_tokenizer(args):
    base_model_id = "meta-llama/Meta-Llama-3-8B"  # Correct Llama 3 model ID
    print(f"Loading base model: {base_model_id}")
    
    if hasattr(args, 'device_map'):
        device_map = args.device_map
    else:
        device_map = 'auto'
    
    # Load base model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id, use_auth_token=True)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map=device_map,
        low_cpu_mem_usage=True,
        use_auth_token=True
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=base_model,
    )
    
    tokenizer.padding_side = 'left'
    
    # Load your LoRA adapter
    if args.pretrained_path:
        model = PeftModel.from_pretrained(base_model, args.pretrained_path, device_map=device_map)
        return model, tokenizer
    
    return base_model, tokenizer