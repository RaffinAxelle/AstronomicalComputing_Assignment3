import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm  # For calculating uncertainties

# Load the FITS file
fits_file = 'nihao_uhd_simulation_g8.26e11_xyz_positions_and_oxygen_ao.fits'
with fits.open(fits_file) as hdul:
    # Assuming the data is in the first Binary Table HDU (index 1)
    data = hdul[1].data  # This will give you a FITS_rec object

# Convert FITS_rec to a NumPy array
numpy_array = np.array(data)

# Initialize lists for A_O and RGal
A_O_list = []
RGal_list = []

# Use a single loop to extract A_O and calculate RGal
for i in range(len(numpy_array)):  # Loop through each row
    A_O_list.append(numpy_array[i][3])  # Access the 4th element (index 3)
    x = numpy_array[i][0]  # First column
    y = numpy_array[i][1]  # Second column
    z = numpy_array[i][2]  # Third column
    RGal_list.append(np.sqrt(x**2 + y**2 + z**2))  # Calculate RGal

# Convert lists to NumPy arrays
A_O = np.array(A_O_list)
RGal = np.array(RGal_list)

# Calculate density for coloring using a 2D histogram with fewer bins for better density representation
hist, xedges, yedges = np.histogram2d(RGal, A_O, bins=1000)

# Use np.digitize to find bin indices for RGal and A_O
x_indices = np.digitize(RGal, xedges) - 1  # Subtracting 1 for zero-based indexing
y_indices = np.digitize(A_O, yedges) - 1

# Clip indices to avoid out-of-bounds errors
x_indices = np.clip(x_indices, 0, hist.shape[0] - 1)
y_indices = np.clip(y_indices, 0, hist.shape[1] - 1)

# Create an array of densities for each point based on histogram counts
density_values = hist[x_indices, y_indices]

# Plotting: Create a 2-panel figure with color mapping for density
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Logarithmic density plot of A_O vs. RGal with color mapping based on density
sc = axs[0].scatter(RGal, A_O, c=density_values, norm=LogNorm(), cmap='viridis', alpha=0.5, s=1)
axs[0].set_xlabel('RGal (kpc)')
axs[0].set_ylabel('A(O)')
axs[0].set_title('Logarithmic Density Plot of RGal vs A(O)')
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
    f'Linear Fit: $A(O) = ({slope:.4f} \pm {slope_std_err:.4f}) \cdot RGal + '
    f'({intercept:.4f} \pm {intercept_std_err:.4f})$'
)
axs[0].legend([fit_equation_label])

# (b) Residuals of the fit: RGal vs ΔA(O)
residuals = A_O - y_pred
axs[1].scatter(RGal, residuals, alpha=0.2, s=1)
axs[1].axhline(0, color='red', linestyle='--', label='Zero Residual')
axs[1].set_xscale('log')
axs[1].set_xlabel('RGal (kpc)')
axs[1].set_ylabel('ΔA(O)')
axs[1].set_title('Residuals of Fit: RGal vs ΔA(O)')
axs[1].legend()

plt.tight_layout()
plt.savefig('figures/log_density_plot.png', dpi=200)
plt.show()

# Report intercept and slope with uncertainties in console output as well.
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")
print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")

# Calculate RMSE for goodness of fit
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}")