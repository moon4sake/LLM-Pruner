import torch


def update_proportion(self, current_prop, losses):
    """Update the proportion of each domain"""
    diff = torch.tensor(losses) - torch.tensor(self.target_loss)
    eta = 1.0
    c = 1e-4  # following Doremi (Xie et al., 2023)

    updated_alpha = torch.log(torch.tensor(current_prop)) + eta * diff
    updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
    updated_domain_weights = (1 - c) * updated_alpha + c / self.n_domains

    updated_domain_weights = updated_domain_weights.numpy().astype("float64")
    updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
    return updated_domain_weights.tolist()


# def update_proportion(self, current_prop, losses):
#     """ Update the proportion of each domain """
#     diff = torch.tensor(losses) - torch.tensor(self.target_loss)
#     eta = 1.
#     c = 1e-4 # following Doremi (Xie et al., 2023)

#     if self.update_type == "doremi": # update with exponential descent
#         updated_alpha = torch.log(torch.tensor(current_prop)) + eta * diff
#         updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
#         updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
#     elif self.update_type == "bandit":
#         updated_alpha = torch.tensor(current_prop) + eta * diff
#         updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
#         updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
#     elif self.update_type == "constant": # constant proportion
#         updated_domain_weights = torch.tensor(current_prop)

#     updated_domain_weights = updated_domain_weights.numpy().astype('float64')
#     updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
#     return updated_domain_weights.tolist()
