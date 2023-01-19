from src.models.model import init_model
import torch

def get_label(input):
    return "cat" if input == 0 else "dog"

def get_prediction(input):
    with torch.no_grad():
        input = input.unsqueeze(0)
        model = init_model(pretrain=True)
        model.eval()
        output = model(input)
        prob = torch.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        top_class = top_class.cpu().detach().numpy()[0][0]
        top_p = top_p.cpu().detach().numpy()[0][0]

    return  f"The image containes a {get_label(top_class)} with {top_p} probability"
