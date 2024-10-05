import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import binned_statistic_2d


# Load the FITS file
fits_file = 'nihao_uhd_simulation_g8.26e11_xyz_positions_and_oxygen_ao.fits'
with fits.open(fits_file) as hdul:
    data = hdul[1].data  # This gives a FITS_rec object

# Convert FITS_rec to a NumPy array
numpy_array = np.array(data)

# Initialize lists for A_O and RGal
A_O_list = []
RGal_list = []
x_list = []
y_list = []

# Use a loop to extract A_O and calculate RGal
for i in range(len(numpy_array)):
    A_O_list.append(numpy_array[i][3])
    x = numpy_array[i][0]
    y = numpy_array[i][1]
    z = numpy_array[i][2]
    x_list.append(x)
    y_list.append(y)
    RGal_list.append(np.sqrt(x**2 + y**2 + z**2))  # Calculate RGal

# Convert lists to NumPy arrays
A_O = np.array(A_O_list)
RGal = np.array(RGal_list)
x_array = np.array(x_list)
y_array = np.array(y_list)

# Calculate density for coloring using a 2D histogram
num_bins = 1000
hist, xedges, yedges = np.histogram2d(RGal, A_O, bins=num_bins)

# Use np.digitize to find bin indices for RGal and A_O
x_indices = np.digitize(RGal, xedges) - 1  # Subtracting 1 for zero-based indexing
y_indices = np.digitize(A_O, yedges) - 1

# Clip indices to avoid out-of-bounds errors
x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)

# Create an array of densities for each point based on histogram counts
density_values = hist[x_indices, y_indices]

# Create a 2-panel figure for the plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Logarithmic density plot of A_O vs. RGal with color mapping based on density
sc = axs[0].scatter(RGal, A_O, c=density_values, norm=LogNorm(), cmap='viridis', alpha=0.5, s=1)
axs[0].set_xlabel('RGal (kpc)')
axs[0].set_ylabel('A(O)')
axs[0].set_title('Logarithmic Density Plot of RGal vs A(O), bins = '+str(num_bins))
plt.colorbar(sc, ax=axs[0], label='Density (log scale)')

# Fit a linear model to the data using sklearn
X = RGal.reshape(-1, 1)  # Reshape for sklearn
y = A_O
model = LinearRegression()  # Python tool to fit a linear function to the data
model.fit(X, y)

# Get predictions and plot linear fit line
y_pred = model.predict(X)
axs[0].plot(RGal, y_pred, color='red', label='Linear Fit')

# Calculate uncertainties using statsmodels for better reporting
X_sm = sm.add_constant(X)  # Add constant term for intercept calculation
model_sm = sm.OLS(y, X_sm).fit()  # Fit using statsmodels

intercept = model_sm.params[0]
slope = model_sm.params[1]
intercept_std_err = model_sm.bse[0]
slope_std_err = model_sm.bse[1]

# Add equation of fit with uncertainties to legend
fit_equation_label = (
    f'Linear Fit: $A(O) = ({slope:.5f} \pm {slope_std_err:.5f}) \cdot RGal + '
    f'({intercept:.4f} \pm {intercept_std_err:.4f})$'
)
axs[0].legend([fit_equation_label])

# Residuals of the fit: ΔA(O) vs RGal
residuals = A_O - y_pred
axs[1].scatter(RGal, residuals, alpha=0.2, s=1)
axs[1].axhline(0, color='red', linestyle='--', label='Zero Residual')
axs[1].set_xscale('log')
axs[1].set_xlabel('RGal (kpc)')
axs[1].set_ylabel('ΔA(O)')
axs[1].set_title('Residuals of Fit: RGal vs ΔA(O), bins = '+str(num_bins))
axs[1].legend()

plt.tight_layout()
plt.savefig('figures/log_density_plot.png', dpi=200)
# plt.show()

# Report intercept and slope with uncertainties in console output as well.
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")
print(f"Slope: {slope:.5f} ± {slope_std_err:.5f}")  # 5 significative numbers because the uncertainty on the slope is really small

# Calculate RMSE for goodness of fit
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}")

# Filter for RGal > 10 kpc: region of large residuals
mask = RGal > 10  # Create a boolean mask
filtered_RGal = RGal[mask]  # Apply the mask to RGal
filtered_A_O = A_O[mask]  # Apply the mask to A_O
filtered_y_pred = y_pred[mask]  # Apply the mask to predicted values

# Calculate RMSE for the filtered data
rmse_filtered = np.sqrt(mean_squared_error(filtered_A_O, filtered_y_pred))

# Print the RMSE for RGal > 10 kpc
print(f"Root Mean Squared Error for RGal > 10 kpc: {rmse_filtered:.4f}")


fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)
num_bins = 1000

# First panel: 2D histogram (binned stats) colored by median simulated A(O)
bin_stat_mean, xedges, yedges, binnumber = binned_statistic_2d(x_array, y_array, A_O, statistic='median', bins=num_bins)
im = axs[0].imshow(bin_stat_mean.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='RdYlBu')
axs[0].set_title("Median simulated A(O), bins = "+str(num_bins))
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
plt.colorbar(im, ax=axs[0], label="Median of A(O)", orientation='horizontal')

# Second panel: 2D histogram (binned stats) colored by median fitted A(O)
bin_stat_mean, xedges, yedges, binnumber = binned_statistic_2d(x_array, y_array, y_pred, statistic='median', bins=num_bins)
im = axs[1].imshow(bin_stat_mean.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='RdYlBu')
axs[1].set_title("Median fitted A(O), bins = "+str(num_bins))
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
plt.colorbar(im, ax=axs[1], label="Median of A(O)", orientation='horizontal')

# Third panel: 2D histogram (binned stats) colored by median residuals A(O)
bin_stat_mean, xedges, yedges, binnumber = binned_statistic_2d(x_array, y_array, residuals, statistic='median', bins=num_bins)
im = axs[2].imshow(bin_stat_mean.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='RdYlBu')
axs[2].set_title("Median residuals A(O), bins = "+str(num_bins))
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
plt.colorbar(im, ax=axs[2], label="Median of A(O)", orientation='horizontal')

# Show plot
plt.tight_layout()
plt.savefig('figures/3_panel_histograms_'+str(num_bins)+'.png', dpi=200)
plt.show()
