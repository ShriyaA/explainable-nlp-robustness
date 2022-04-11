from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
from attribution.attribution import Attribution
import torch

class SaliencyMapVisualizer():

    def __init__(self, model, tokenizer, device, expl_method) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.expl_method = expl_method
        self.attributor = Attribution(model, tokenizer, device, expl_method)

    def visualize_attribution(self, text, true_label):
        input_ids, ref_input_ids = self.attributor.construct_input_ref_pair(text, self.tokenizer, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id)
        position_ids, ref_position_ids = self.attributor.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.attributor.construct_attention_mask(input_ids)
        scores = self.attributor.predict(input_ids, position_ids, attention_mask)
        #print(scores)
        indices = input_ids[0].detach().tolist()
        all_tokens = self.attributor.tokenizer.convert_ids_to_tokens(indices)
        expl_method = self.attributor.resolve_expl_method()

        expl = expl_method(self.attributor.forward_func, self.model.roberta.embeddings.word_embeddings)
        attributions, delta = expl.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(position_ids, attention_mask),
                                        return_convergence_delta=True)
        attributions_sum = self.attributor.summarize_attributions(attributions)

        sentence_vis = viz.VisualizationDataRecord(
                attributions_sum,
                torch.max(torch.softmax(scores[0], dim=0)),
                torch.argmax(scores),
                true_label,
                text,
                attributions_sum.sum(),
                all_tokens,
                delta)
        return (viz.visualize_text([sentence_vis]), attributions_sum.detach().tolist(), all_tokens)
