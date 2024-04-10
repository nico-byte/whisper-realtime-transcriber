import math
import torch

from alive_progress import alive_bar


def dummy_training(device, dtype, learning_rate, timesteps, x, y):
    # Randomly initialize weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    for _ in range(timesteps):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        yield


def set_device():
    # check the pytorch version
    print(f'PyTorch version: {torch.__version__}')
    # this ensures that cuda
    print(f'CUDA is available: {torch.cuda.is_available()}')
    # for metal device
    print('Check for metal device...')
    # this ensures that the current macOS version is at least 12.3+
    print(f'MPS is available: {torch.backends.mps.is_available()}')
    # this ensures that the current PyTorch installation was built with MPS activated.
    print(f'PyTorch was built with MPS enabled: {torch.backends.mps.is_built()}')

    dtype = torch.float
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Device {device} is hooked.')
    # Create random input and output data
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    learning_rate = 1e-6
    timesteps = 2000

    with alive_bar(timesteps, title='Dummy Training', force_tty=True, ctrl_c=False, bar='filling') as bar:
        for _ in dummy_training(device, dtype, learning_rate, timesteps, x, y):
            # time.sleep(0.00002)
            bar()

    print('No Errors, you are good to go!')

    return device
