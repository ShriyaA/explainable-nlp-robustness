from captum.attr import LayerIntegratedGradients, Saliency, InputXGradient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Attribution():

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
        if self.expl_method == 'Gradient':
            return Saliency
        if self.expl_method == 'InputXGradient':
            return InputXGradient

    def get_attribution(self, text, true_label, word_level=True, combination_method='avg'):
        
        input_ids, ref_input_ids = self.construct_input_ref_pair(text, self.tokenizer, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)

        scores = self.predict(input_ids, position_ids, attention_mask)

        indices = input_ids[0].detach().tolist()
        expl_method = self.resolve_expl_method()

        expl = expl_method(self.forward_func, self.model.roberta.embeddings.word_embeddings)
        attributions, delta = expl.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(position_ids, attention_mask),
                                        return_convergence_delta=True)
        attributions_sum = self.summarize_attributions(attributions)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)[1:][:-1]
        attributions_sum = attributions_sum[1:][:-1]
        if word_level:
            attributions_sum, tokens = self.word_level_attribution(tokens, attributions_sum.tolist(), combination_method)
        return attributions_sum, [x[1:] if x[0] == 'Ġ' else x for x in tokens], torch.argmax(scores)

    def word_level_attribution(self, tokens, token_attr_scores, combination_method='avg'):
        '''
        return:
        list of tuples where each tuple is of the format (word_level_score, index_list, word).
        word_level_score:   Score of the word calculated by adding scores of all the tokens.
        index_list      :   List of indexes of the tokens that form a word.
        word            :   Complete word created by appending tokens.
        '''
        # Convert tokens to words by joining and calc. score by adding scores of all the tokens in a word
        score_word_list = []
        word_list = []
        curr_list = []
        curr_scores = []
        curr_str = ""
        for i in range(len(tokens)):
            if (tokens[i][0] == 'Ġ' and len(curr_list) != 0):
                if combination_method == 'sum':
                    curr_score = sum(curr_scores)
                elif combination_method == 'max':
                    curr_score = max(curr_scores)
                else:
                    curr_score = sum(curr_scores)/len(curr_scores)
                score_word_list.append(curr_score)
                word_list.append(curr_str)
                curr_scores = []
                curr_list = []
                curr_str = ""

            curr_scores.append(token_attr_scores[i])
            curr_list.append(i)
            curr_str += tokens[i]
        if combination_method == 'sum':
            curr_score = sum(curr_scores)
        elif combination_method == 'max':
            curr_score = max(curr_scores)
        else:
            curr_score = sum(curr_scores)/len(curr_scores)
        score_word_list.append(curr_score)
        word_list.append(curr_str)
        return torch.tensor(score_word_list, dtype=torch.float64), word_list


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "textattack/roberta-base-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.zero_grad()

    text = "Culpable politicians moderately downsize"

    attributor = Attribution(model, tokenizer, device, "IntegratedGradients")
    print(attributor.get_attribution(text, 1, word_level=True))
