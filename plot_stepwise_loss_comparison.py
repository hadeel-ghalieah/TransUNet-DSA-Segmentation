import numpy as np
import matplotlib.pyplot as plt

losses_transunet = np.load("stepwise_losses_transunet.npy")
losses_dsa = np.load("stepwise_losses_dsa_transunet.npy")

def smooth_curve(data, window_size=20):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

smoothed_transunet = smooth_curve(losses_transunet)
smoothed_dsa = smooth_curve(losses_dsa)

plt.figure(figsize=(14, 6))
plt.plot(smoothed_transunet, label="TransUNet (Smoothed)", color='blue', linewidth=1.5)
plt.plot(smoothed_dsa, label="DSA_TransUNet (Smoothed)", color='red', linewidth=1.5)
plt.title("Stepwise Loss Comparison Between TransUNet and DSA_TransUNet")
plt.xlabel("Training Steps (Batches)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("stepwise_loss_comparison.png")
plt.show()

# Compare final average loss
avg_loss_transunet = np.mean(losses_transunet[-50:])
avg_loss_dsa = np.mean(losses_dsa[-50:])

print("\n📊 Final 50-step Average Loss:")
print(f"TransUNet: {avg_loss_transunet:.6f}")
print(f"DSA_TransUNet: {avg_loss_dsa:.6f}")

if avg_loss_transunet < avg_loss_dsa:
    print("\n✅ TransUNet performs better based on recent loss.")
else:
    print("\n✅ DSA_TransUNet performs better based on recent loss.")
