import os

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()
# sns.set_context("talk")
# sns.set_style('white')

STATS_DIR = 'logs/statistics'
STAT_FILENAMES = [f for f in os.listdir(STATS_DIR) if f!='.gitignore']
if not os.path.exists('plots'):
    os.makedirs('plots')


for f in STAT_FILENAMES:
    print(f)
    fname = f[:-4]
    shorter_fname = '_'.join(fname.split('_')[2:6])

    df = pd.read_csv(os.path.join(STATS_DIR, f))

    fig = plt.figure()

    ax = fig.add_subplot(211)
    df.plot(
        y='Loss_G',
        title='Training Losses',
        ax=ax)
    if 'adv0_' not in shorter_fname:
        df.plot(
            y='Loss_D',
            secondary_y=True,
            ax=ax)

    ax = fig.add_subplot(212)
    df.plot(
        ax=ax,
        y='PSNR',
        title='Validation Metrics')
    df.plot(
        y='SSIM',
        secondary_y=True,
        ax=ax)

    fig.suptitle(shorter_fname, size=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join('plots', shorter_fname+'.png'))
