from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
import torch

class SaliencyMapVisualizer():

    def __init__(self, model, tokenizer, device, expl_method) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.expl_method = expl_method

    def predict(self, inputs, position_ids=None, attention_mask=None):
        output = self.model(inputs, position_ids=position_ids, attention_mask=attention_mask)
        return output.logits

    def forward_func(self, inputs, position_ids=None, attention_mask=None):
        pred = self.predict(inputs, position_ids=position_ids, attention_mask=attention_mask)
        return pred.max(1).values

    def construct_input_ref_pair(self, text, tokenizer, ref_token_id, sep_token_id, cls_token_id):
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        # construct input token ids
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        # construct reference token ids 
        ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

        return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device)

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(start=2, end=seq_length+2, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids

    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions 

    def resolve_expl_method(self):
        if self.expl_method == 'IntegratedGradients':
            return LayerIntegratedGradients

    def get_attribution(self, text, true_label):
        input_ids, ref_input_ids = self.construct_input_ref_pair(text, self.tokenizer, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        expl_method = self.resolve_expl_method()

        expl = expl_method(self.forward_func, self.model.roberta.embeddings.word_embeddings)
        attributions, delta = expl.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(position_ids, attention_mask),
                                        return_convergence_delta=True)
        attributions_sum = self.summarize_attributions(attributions)
        return (indices, attributions_sum, true_label)

    def visualize_attribution(self, text, true_label):
        input_ids, ref_input_ids = self.construct_input_ref_pair(text, self.tokenizer, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)
        scores = self.predict(input_ids, position_ids, attention_mask)
        #print(scores)
        indices = input_ids[0].detach().tolist()
        all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
        expl_method = self.resolve_expl_method()

        expl = expl_method(self.forward_func, self.model.roberta.embeddings.word_embeddings)
        attributions, delta = expl.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(position_ids, attention_mask),
                                        return_convergence_delta=True)
        attributions_sum = self.summarize_attributions(attributions)

        sentence_vis = viz.VisualizationDataRecord(
                attributions_sum,
                torch.max(torch.softmax(scores[0], dim=0)),
                torch.argmax(scores),
                true_label,
                text,
                attributions_sum.sum(),
                all_tokens,
                delta)
        return viz.visualize_text([sentence_vis])