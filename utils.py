import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.dataset import tensorFromSentence
from transformer.transformer import create_look_ahead_mask

__SOS_IMAGE_TOKEN__  = 64
__EOS_IMAGE_TOKEN__  = 65
__MASK_IMAGE_TOKEN__ = 66

__SOS_TEXT_TOKEN__   = 67
__EOS_TEXT_TOKEN__   = 68
__PAD_TEXT_TOKEN__   = 69


def number_of_parameters(model: nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of parameters: {}".format(total))
    print("Trainable number of parameters: {}".format(trainable))



def generate_from_text(text, train_dataset, transformer, vqvae, temperature):

	max_len = train_dataset.max_text_length

	text = text.lower()
	text_as_tensor = tensorFromSentence(
		train_dataset.annotations_language, 
		train_dataset.bpe_tokenizer.process_line(text), 
		max_len
	)

	text_len  = text_as_tensor.shape[0]
	total_len = transformer.seq_len_image + transformer.seq_len_text 

	in_ = torch.full((1, total_len), __PAD_TEXT_TOKEN__)
	in_[:, :max_len  ] = text_as_tensor
	in_[:, max_len   ] = __SOS_IMAGE_TOKEN__
	in_[:, max_len+1:] = __MASK_IMAGE_TOKEN__
	in_ = in_.to(vqvae.device) 

	mask = create_look_ahead_mask(in_)

	for i in range(transformer.seq_len_image - 2):
		with torch.no_grad(): logits = transformer(in_, mask)
		next_tokens = torch.multinomial(
			F.softmax(logits[:, max_len + i] / temperature, dim=-1), 1)
		in_[:, max_len + i + 1] = next_tokens[:, 0]

	model_picture = in_[:, max_len:][:, 1:-1].view(-1, 16, 16)
	model_picture = torch.minimum(model_picture, torch.full_like(model_picture, 63))

	decoded_picture = vqvae.decode_code(model_picture)[0].cpu().detach().permute(1, 2, 0)
	return decoded_picture.cpu()


