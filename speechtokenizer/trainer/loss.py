import torch
import torchaudio
import matplotlib.pylab as plt

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_transform = torchaudio.transforms.MelScale(n_mels=num_mels, sample_rate=sample_rate, n_stft=n_fft//2+1, f_min=fmin, f_max=fmax, norm='slaney', mel_scale="htk")
        mel_basis[str(fmax)+'_'+str(y.device)] = mel_transform.fb.float().T.to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.abs(spec) + 1e-9
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def recon_loss(x, x_hat):
    length = min(x.size(-1), x_hat.size(-1))
    return torch.nn.functional.l1_loss(x[:, :, :length], x_hat[:, :, :length])

def mel_loss(x, x_hat, **kwargs):
    x_mel = mel_spectrogram(x.squeeze(1), **kwargs)
    x_hat_mel = mel_spectrogram(x_hat.squeeze(1), **kwargs)
    length = min(x_mel.size(2), x_hat_mel.size(2))
    return torch.nn.functional.l1_loss(x_mel[:, :, :length], x_hat_mel[:, :, :length])

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)

    return loss


def adversarial_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l

    return loss


def d_axis_distill_loss(feature, target_feature):
    n = min(feature.size(1), target_feature.size(1))
    distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    n = min(feature.size(1), target_feature.size(1))
    l1_loss = torch.functional.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
    sim_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
    distill_loss = l1_loss + lambda_sim * sim_loss
    return distill_loss 