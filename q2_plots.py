import numpy as np
import matplotlib.pyplot as plt
from astropy.io.votable import parse_single_table
from astropy.coordinates import Distance
import astropy.units as u
import os

# Read the VOTable file
table = parse_single_table("stars_filtered-result.vot").to_table()

# Convert parallax to mas and create a Quantity object
parallax = u.Quantity(table['parallax'], u.mas)

# Calculate distance
distance = Distance(parallax=parallax)

# Calculate absolute G magnitude
abs_g_mag = table['phot_g_mean_mag'] - 5 * np.log10(distance.pc) + 5

# Create the figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Panel (a): Gaia BP-RP vs. absolute G magnitude
bp_rp = table['phot_bp_mean_mag'] - table['phot_rp_mean_mag']
ax1.scatter(bp_rp, abs_g_mag, alpha=0.5, s=3, color='m')
ax1.set_xlabel('BP - RP')
ax1.set_ylabel('Absolute G Magnitude')
ax1.set_title('Gaia CMD')
ax1.invert_yaxis()

# Panel (b): 2MASS J-Ks vs. apparent K magnitude
j_ks = table['j_m'] - table['ks_m']
ax2.scatter(j_ks, table['ks_m'], alpha=0.5, s=3, color='m')
ax2.set_xlabel('J - Ks')
ax2.set_ylabel('Apparent K Magnitude')
ax2.set_title('2MASS CMD')
ax2.invert_yaxis()

# Adjust layout and save
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/cmds_M67.png', dpi=200)
plt.show()

# Print statistics
print(f"Total stars in the data: {len(table)}")

# Recommendation
if len(table) >= 392:  # 2dF has 392 science fibres
    recommendation = "There are enough bright stars for a full 2dF configuration."
else:
    recommendation = f"There are only {len(table)} suitable stars, which is not enough for a full 2dF configuration."

print(f"\nRecommendation: {recommendation}")