def train(model, device, train_loader, optimizer, criterion, epoch):
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0

	for data, target in train_loader:
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * data.size(0)
		pred = output.argmax(dim=1)
		correct += pred.eq(target).sum().item()
		total += data.size(0)

	print(
		f"Epoch {epoch}: train loss={total_loss / total:.4f}, "
		f"train acc={100.0 * correct / total:.2f}%"
	)
	return total_loss / total, 100.0 * correct / total
