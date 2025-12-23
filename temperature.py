import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.widgets import Slider

# Initial percentage probabilities and tokens
initial_probs = np.array([0.45, 0.3, 0.15, 0.09, 0.01])  # Must sum to 1
tokens = ["pizza", "chocolate", "burgers", "pasta", "pencil"]

# Convert probabilities to logits (inverse softmax up to a constant)
def probs_to_logits(probs):
    return np.log(probs)

logits = probs_to_logits(initial_probs)

# Softmax function with temperature (handles T = 0 as greedy decoding)
def softmax_with_temp(logits, temperature):
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

# Initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
initial_temp = 1.0
probs = softmax_with_temp(logits, initial_temp)
bars = ax.bar(tokens, probs, color='orange')

ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title(f"Temperature = {initial_temp}", fontweight='bold')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))  # Add ticks every 10%

# Slider setup
ax_temp = plt.axes([0.2, 0.1, 0.65, 0.03])
temp_slider = Slider(ax_temp, 'Temperature', 0.0, 2.5, valinit=initial_temp, valstep=0.1)

# Update function
def update(val):
    temp = temp_slider.val
    new_probs = softmax_with_temp(logits, temp)
    for bar, prob in zip(bars, new_probs):
        bar.set_height(prob)
    ax.set_title(f"Temperature = {temp:.1f}", fontweight='bold')
    fig.canvas.draw_idle()

# Connect slider to update function
temp_slider.on_changed(update)

plt.show()