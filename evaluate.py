import torch


@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	for data, target in test_loader:
		data, target = data.to(device), target.to(device)
		output = model(data)
		loss = criterion(output, target)

		total_loss += loss.item() * data.size(0)
		pred = output.argmax(dim=1)
		correct += pred.eq(target).sum().item()
		total += data.size(0)

	avg_loss = total_loss / total
	accuracy = 100.0 * correct / total
	print(f"Test loss={avg_loss:.4f}, test acc={accuracy:.2f}%")
	return avg_loss, accuracy
