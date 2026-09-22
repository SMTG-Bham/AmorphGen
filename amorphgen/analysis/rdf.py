"""Pair distribution function g(r) and structure factor S(q).

compute_rdf                      total / partial g(r), ensemble-averaged
compute_averaged_rdf             ensemble-mean g(r)
compute_structure_factor_direct  S(q) by Debye sum at reciprocal-lattice q
compute_structure_factor         S(q) by Fourier transform of g(r); g(r) is
                                 cut at L/2, which damps the FSDP
xray_form_factor                 f0(q), Waasmaier-Kirfel (1995)

``weighting``: "xray" (q-dependent f0(q)), "neutron" (Varley, F. Sears. "Neutron scattering lengths and cross sectioirn." Neutron news 3.3 (1992): 29-37.), or
"unweighted" (f = 1). Multi-component S(q) is the Faber-Ziman sum of partials,
w_ab = (2 - delta_ab) c_a c_b f_a f_b / <f>^2, with partials using the total
number density. Conventions: Keen, David A. "A comparison of various commonly used correlation functions for describing total scattering." Applied Crystallography 34.2 (2001): 172-177.
"""

from __future__ import annotations

import numpy as np
from ase.neighborlist import neighbor_list

from .cutoff import check_rmax

# Default Gaussian smearing (A) applied to g(r)
# the thermal/experimental broadening, so simulated peaks compare naturally
# with diffraction data. Pass sigma=0.0 for the raw histogram. S(q) is ALWAYS
# computed from the raw g(r) (smearing would damp it by exp(-q^2 sigma^2 / 2)).
DEFAULT_SMEARING = 0.05

# Default q re-binning width (1/A) applied by the CLI / plots to the DIRECT
# S(q) for presentation (n_per_bin-weighted Gaussian, see
# _smooth_sq_weighted). Library calls default to 0 (raw) so API users get the
# exact shell averages; 0.05 halves the low-q speckle for ~5% of the FSDP.
DEFAULT_SQ_SMOOTH = 0.05


def _gaussian_smear(r, g_r, sigma):
    """Apply Gaussian broadening to g(r)."""
    if sigma <= 0:
        return g_r
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    window_size = int(4 * sigma / dr)  # 4-sigma window
    if window_size < 1:
        return g_r
    x = np.arange(-window_size, window_size + 1) * dr
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(g_r, kernel, mode='same')


def compute_rdf(atoms_list, pair=None, rmax=None, nbins=200,
                sigma=DEFAULT_SMEARING):
    """Compute the radial distribution function g(r).

    Parameters
    ----------
    atoms_list : list of Atoms
    pair : str, optional
        Pair to analyse, e.g. "Si-O". If None, total RDF.
    rmax : float, optional
        Maximum radius in A. Auto-detected from cell if None.
    nbins : int
        Number of histogram bins (default 200).
    sigma : float
        Gaussian smearing width in A (default ``DEFAULT_SMEARING`` = 0.05,
        comparable to thermal/experimental broadening). Peak positions are
        unchanged; heights drop and widths grow. Pass 0.0 for the raw
        histogram.
    """
    if not atoms_list:
        return {"r": [], "g_r": []}
    if rmax is None:
        half_cells = [min(a.cell.lengths()) / 2 for a in atoms_list]
        rmax = float(np.floor(min(half_cells) * 10) / 10)
    check_rmax(atoms_list, rmax)

    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)
    shell_vols = 4 * np.pi * r_centres**2 * dr

    g_r = np.zeros(nbins)
    n_frames = len(atoms_list)

    for atoms in atoms_list:
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
        syms = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()
        n = len(atoms)

        if pair is not None:
            p1, p2 = pair.split("-")
            n_source = int(np.sum(syms == p1))
            n_target = int(np.sum(syms == p2))
            if p1 == p2:
                rho_target = (n_target - 1) / vol
            else:
                rho_target = n_target / vol
            mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)
            pair_dists = dists[mask]
        else:
            n_source = n
            rho_target = (n - 1) / vol
            pair_dists = dists

        if len(pair_dists) == 0 or n_source == 0 or rho_target <= 0:
            continue

        # Use proper histogram — skip distances outside [0, rmax]
        in_range = (pair_dists > 0) & (pair_dists < rmax)
        if not np.any(in_range):
            continue
        hist, _ = np.histogram(pair_dists[in_range], bins=nbins,
                               range=(0, rmax))

        valid = shell_vols > 0
        g_r[valid] += hist[valid] / (n_source * rho_target * shell_vols[valid])

    g_r /= n_frames

    if sigma > 0:
        g_r = _gaussian_smear(r_centres, g_r, sigma)

    return {"r": r_centres.tolist(), "g_r": g_r.tolist()}


# Neutron coherent scattering lengths b_c (fm) — common amorphous-system
# elements. Bound coherent values (real part) from V. F. Sears, Neutron News
# 3(3), 26-37 (1992), Table 1; every entry checked against the printed table.
# Used when weighting="neutron".
_NEUTRON_B = {
    "H": -3.7390, "D":  6.671,  "Li": -1.90,  "Be":  7.79,  "B":   5.30,
    "C":  6.6460, "N":  9.36,   "O":  5.803,  "F":   5.654, "Na":  3.63,
    "Mg": 5.375,  "Al": 3.449,  "Si": 4.1491, "P":   5.13,  "S":   2.847,
    "Cl": 9.5770, "K":  3.67,   "Ca": 4.70,   "Ti":  -3.438,"V":  -0.3824,
    "Cr": 3.635,  "Mn": -3.73,  "Fe": 9.45,   "Co":  2.49,  "Ni": 10.3,
    "Cu": 7.718,  "Zn": 5.680,  "Ga": 7.288,  "Ge":  8.185, "As":  6.58,
    "Se": 7.970,  "Br": 6.795,  "Y":  7.75,   "Zr":  7.16,  "Nb":  7.054,
    "Mo": 6.715,  "Pd": 5.91,   "Ag": 5.922,  "Cd":  4.87,  "In":  4.065,
    "Sn": 6.225,  "Sb": 5.57,   "Te": 5.80,   "I":   5.28,  "Cs":  5.42,
    "Ba": 5.07,   "La": 8.24,   "Ce": 4.84,   "Hf":  7.77,  "Ta":  6.91,
    "W":  4.86,   "Pt": 9.6,    "Au": 7.63,   "Pb":  9.405, "Bi":  8.532,
}

# X-ray atomic form factors f0(q): 5-Gaussian parameterisation
#   f0(q) = sum_{i=1..5} a_i * exp(-b_i * s^2) + c,   s = q / (4*pi) = sin(theta)/lambda
# Coefficients: D. Waasmaier & A. Kirfel, Acta Cryst. A51, 416-431 (1995),
# Table 1(a), valid to s = 6 A^-1 (q ~ 75 A^-1). Validated: f0(0) = Z for
# all 98 elements; agrees with the International Tables for Crystallography
# Vol. C (Table 6.1.1.4) Cromer-Mann set to < 0.5 % at q = 2.45 A^-1 for
# the oxide/semiconductor elements; N, O, F, Ni, Cu, Zn, Ga, Ge, As checked
# digit-for-digit against the printed table.
# Entry: symbol -> ((a1..a5), (b1..b5), c)
_WK95_XRAY = {
    "H": ((0.493000, 0.322912, 0.140191, 0.040810, 0.000000),
        (10.510900, 26.125700, 3.142360, 57.799700, 0.000000), 0.003038),
    "He": ((0.732354, 0.753896, 0.283819, 0.190003, 0.039139),
         (11.553918, 4.595831, 1.546299, 26.463964, 0.377523), 0.000487),
    "Li": ((0.974637, 0.158472, 0.811855, 0.262416, 0.790108),
         (4.334946, 0.342451, 97.102969, 201.363824, 1.409234), 0.002542),
    "Be": ((1.533712, 0.638283, 0.601052, 0.106139, 1.118414),
         (42.662078, 0.595420, 99.106501, 0.151340, 1.843093), 0.002511),
    "B": ((2.085185, 1.064580, 1.062788, 0.140515, 0.641784),
        (23.494069, 1.137894, 61.238975, 0.114886, 0.399036), 0.003823),
    "C": ((2.657506, 1.078079, 1.490909, -4.241070, 0.713791),
        (14.780758, 0.776775, 42.085843, -0.000294, 0.239535), 4.297963),
    "N": ((11.893780, 3.277479, 1.858092, 0.858927, 0.912985),
        (0.000158, 10.232723, 30.344690, 0.656065, 0.217287), -11.804902),
    "O": ((2.960427, 2.508818, 0.637853, 0.722838, 1.142756),
        (14.182259, 5.936858, 0.112726, 34.958481, 0.390240), 0.027014),
    "F": ((3.511943, 2.772244, 0.678385, 0.915159, 1.089261),
        (10.687859, 4.380466, 0.093982, 27.255203, 0.313066), 0.032557),
    "Ne": ((4.183749, 2.905726, 0.520513, 1.135641, 1.228065),
         (8.175457, 3.252536, 0.063295, 21.813909, 0.224952), 0.025576),
    "Na": ((4.910127, 3.081783, 1.262067, 1.098938, 0.560991),
         (3.281434, 9.119178, 0.102763, 132.013942, 0.405878), 0.079712),
    "Mg": ((4.708971, 1.194814, 1.558157, 1.170413, 3.239403),
         (4.875207, 108.506079, 0.111516, 48.292407, 1.928171), 0.126842),
    "Al": ((4.730796, 2.313951, 1.541980, 1.117564, 3.154754),
         (3.628931, 43.051166, 0.095960, 108.932389, 1.555918), 0.139509),
    "Si": ((5.275329, 3.191038, 1.511514, 1.356849, 2.519114),
         (2.631338, 33.730728, 0.081119, 86.288640, 1.170087), 0.145073),
    "P": ((1.950541, 4.146930, 1.494560, 1.522042, 5.729711),
        (0.908139, 27.044953, 0.071280, 67.520190, 1.981173), 0.155233),
    "S": ((6.372157, 5.154568, 1.473732, 1.635073, 1.209372),
        (1.514347, 22.092528, 0.061373, 55.445176, 0.646925), 0.154722),
    "Cl": ((1.446071, 6.870609, 6.151801, 1.750347, 0.634168),
         (0.052357, 1.193165, 18.343416, 46.398394, 0.401005), 0.146773),
    "Ar": ((7.188004, 6.638454, 0.454180, 1.929593, 1.523654),
         (0.956221, 15.339877, 15.339862, 39.043824, 0.062409), 0.265954),
    "K": ((8.163991, 7.146945, 1.070140, 0.877316, 1.486434),
        (12.816323, 0.808945, 210.327009, 39.597651, 0.052821), 0.253614),
    "Ca": ((8.593655, 1.477324, 1.436254, 1.182839, 7.113258),
         (10.460644, 0.041891, 81.390382, 169.847839, 0.688098), 0.196255),
    "Sc": ((1.476566, 1.487278, 1.600187, 9.177463, 7.099750),
         (53.131022, 0.035325, 137.319495, 9.098031, 0.602102), 0.157765),
    "Ti": ((9.818524, 1.522646, 1.703101, 1.768774, 7.082555),
         (8.001879, 0.029763, 39.885423, 120.158000, 0.532405), 0.102473),
    "V": ((10.473575, 1.547881, 1.986381, 1.865616, 7.056250),
        (7.081940, 0.026040, 31.909672, 108.022844, 0.474882), 0.067744),
    "Cr": ((11.007069, 1.555477, 2.985293, 1.347855, 7.034779),
         (6.366281, 0.023987, 23.244838, 105.774500, 0.429369), 0.065510),
    "Mn": ((11.709542, 1.733414, 2.673141, 2.023368, 7.003180),
         (5.597120, 0.017800, 21.788419, 89.517915, 0.383054), -0.147290),
    "Fe": ((12.311098, 1.876623, 3.066177, 2.070451, 6.975185),
         (5.009415, 0.014461, 18.743041, 82.767874, 0.346506), -0.304931),
    "Co": ((12.914510, 2.481908, 3.466894, 2.106351, 6.960892),
         (4.507138, 0.009126, 16.438130, 76.987317, 0.314418), -0.936572),
    "Ni": ((13.521865, 6.947285, 3.866028, 2.135900, 4.284731),
         (4.077277, 0.286763, 14.622634, 71.966078, 0.004437), -2.762697),
    "Cu": ((14.014192, 4.784577, 5.056806, 1.457971, 6.932996),
         (3.738280, 0.003744, 13.034982, 72.554793, 0.265666), -3.254477),
    "Zn": ((14.741002, 6.907748, 4.642337, 2.191766, 38.424042),
         (3.388232, 0.243315, 11.903689, 63.312130, 0.000397), -36.915828),
    "Ga": ((15.758946, 6.841123, 4.121016, 2.714681, 2.395246),
         (3.121754, 0.226057, 12.482196, 66.203622, 0.007238), -0.847395),
    "Ge": ((16.540614, 1.567900, 3.727829, 3.345098, 6.785079),
         (2.866618, 0.012198, 13.432163, 58.866046, 0.210974), 0.018726),
    "As": ((17.025643, 4.503441, 3.715904, 3.937200, 6.790175),
         (2.597739, 0.003012, 14.272119, 50.437997, 0.193015), -2.984117),
    "Se": ((17.354071, 4.653248, 4.259489, 4.136455, 6.749163),
         (2.349787, 0.002550, 15.579460, 45.181201, 0.177432), -3.160982),
    "Br": ((17.550570, 5.411882, 3.937180, 3.880645, 6.707793),
         (2.119226, 16.557185, 0.002481, 42.164009, 0.162121), -2.492088),
    "Kr": ((17.655279, 6.848105, 4.171004, 3.446760, 6.685200),
         (1.908231, 16.606235, 0.001598, 39.917471, 0.146896), -2.810592),
    "Rb": ((8.123134, 2.138042, 6.761702, 1.156051, 17.679547),
         (15.142385, 33.542666, 0.129372, 224.132506, 1.713368), 1.139548),
    "Sr": ((17.730219, 9.795867, 6.099763, 2.620025, 0.600053),
         (1.563060, 14.310868, 0.120574, 135.771318, 0.120574), 1.140251),
    "Y": ((17.792040, 10.253252, 5.714949, 3.170516, 0.918251),
        (1.429691, 13.132816, 0.112173, 108.197029, 0.112173), 1.131787),
    "Zr": ((17.859771, 10.911038, 5.821111, 3.512513, 0.746965),
         (1.310692, 12.319285, 0.104353, 91.777544, 0.104353), 1.124859),
    "Nb": ((17.958398, 12.063054, 5.007015, 3.287667, 1.531019),
         (1.211590, 12.246687, 0.098615, 75.011944, 0.098615), 1.123452),
    "Mo": ((6.236218, 17.987711, 12.973127, 3.451426, 0.210899),
         (0.090780, 1.108310, 11.468720, 66.684153, 0.090780), 1.108770),
    "Tc": ((17.840964, 3.428236, 1.373012, 12.947364, 6.335469),
         (1.005729, 41.901383, 119.320541, 9.781542, 0.083391), 1.074784),
    "Ru": ((6.271624, 17.906739, 14.123269, 3.746008, 0.908235),
         (0.077040, 0.928222, 9.555345, 35.860678, 123.552247), 1.043992),
    "Rh": ((6.216648, 17.919738, 3.854252, 0.840326, 15.173498),
         (0.070789, 0.856121, 33.889484, 121.686688, 9.029517), 0.995452),
    "Pd": ((6.121511, 4.784063, 16.631683, 4.318258, 13.246773),
         (0.062549, 0.784031, 8.751391, 34.489983, 0.784031), 0.883099),
    "Ag": ((6.073874, 17.155437, 4.173344, 0.852238, 17.988685),
         (0.055333, 7.896512, 28.443739, 110.376108, 0.716809), 0.756603),
    "Cd": ((6.080986, 18.019468, 4.018197, 1.303510, 17.974669),
         (0.048990, 7.273646, 29.119283, 95.831208, 0.661231), 0.603504),
    "In": ((6.196477, 18.816183, 4.050479, 1.638929, 17.962912),
         (0.042072, 6.695665, 31.009791, 103.284350, 0.610714), 0.333097),
    "Sn": ((19.325171, 6.281571, 4.498866, 1.856934, 17.917318),
         (6.118104, 0.036915, 32.529047, 95.037182, 0.565651), 0.119024),
    "Sb": ((5.394956, 6.549570, 19.650681, 1.827820, 17.867833),
         (33.326523, 0.030974, 5.564929, 87.130965, 0.523992), -0.290506),
    "Te": ((6.660302, 6.940756, 19.847015, 1.557175, 17.802427),
         (33.031656, 0.025750, 5.065547, 84.101613, 0.487660), -0.806668),
    "I": ((19.884502, 6.736593, 8.110516, 1.170953, 17.548715),
        (4.628591, 0.027754, 31.849096, 84.406391, 0.463550), -0.448811),
    "Xe": ((19.978920, 11.774945, 9.332182, 1.244749, 17.737501),
         (4.143356, 0.010142, 28.796199, 75.280688, 0.413616), -6.065902),
    "Cs": ((17.418675, 8.314444, 10.323193, 1.383834, 19.876252),
         (0.399828, 0.016872, 25.605828, 233.339674, 3.826915), -2.322802),
    "Ba": ((19.747344, 17.368476, 10.465718, 2.592602, 11.003653),
         (3.481823, 0.371224, 21.226641, 173.834271, 0.010719), -5.183497),
    "La": ((19.966018, 27.329654, 11.018425, 3.086696, 17.335454),
         (3.197408, 0.003446, 19.955492, 141.381979, 0.341817), -21.745489),
    "Ce": ((17.355121, 43.988498, 20.546650, 3.130670, 11.353665),
         (0.328369, 0.002047, 3.088196, 134.907661, 18.832961), -38.386017),
    "Pr": ((21.551311, 17.161729, 11.903859, 2.679103, 9.564197),
         (2.995675, 0.312491, 17.716705, 152.192827, 0.010468), -3.871068),
    "Nd": ((17.331244, 62.783923, 12.160097, 2.663483, 22.239951),
         (0.300269, 0.001320, 17.026001, 148.748986, 2.910268), -57.189844),
    "Pm": ((17.286388, 51.560161, 12.478557, 2.675515, 22.960947),
         (0.286620, 0.001550, 16.223755, 143.984513, 2.796480), -45.973681),
    "Sm": ((23.700364, 23.072215, 12.777782, 2.684217, 17.204366),
         (2.689539, 0.003491, 15.495437, 139.862475, 0.274536), -17.452166),
    "Eu": ((17.186195, 37.156839, 13.103387, 2.707246, 24.419271),
         (0.261678, 0.001995, 14.787360, 134.816293, 2.581880), -31.586687),
    "Gd": ((24.898118, 17.104951, 13.222581, 3.266152, 48.995214),
         (2.435028, 0.246961, 13.996325, 110.863093, 0.001383), -43.505684),
    "Tb": ((25.910013, 32.344139, 13.765117, 2.751404, 17.064405),
         (2.373912, 0.002034, 13.481969, 125.836511, 0.236916), -26.851970),
    "Dy": ((26.671785, 88.687577, 14.065445, 2.768497, 17.067782),
         (2.282593, 0.000665, 12.920230, 121.937188, 0.225531), -83.279831),
    "Ho": ((27.150190, 16.999819, 14.059334, 3.386979, 46.546471),
         (2.169660, 0.215414, 12.213148, 100.506781, 0.001211), -41.165253),
    "Er": ((28.174886, 82.493269, 14.624002, 2.802756, 17.018515),
         (2.121100, 0.000640, 11.915256, 114.529936, 0.207519), -77.135221),
    "Tm": ((28.925894, 76.173796, 14.904704, 2.814812, 16.998117),
         (2.046203, 0.000656, 11.465375, 111.411979, 0.199376), -70.839813),
    "Yb": ((29.676760, 65.624068, 15.160854, 2.830288, 16.997850),
         (1.977630, 0.000720, 11.044622, 108.139150, 0.192110), -60.313812),
    "Lu": ((30.122865, 15.099346, 56.314899, 3.540980, 16.943730),
         (1.883090, 10.342764, 0.000780, 89.559248, 0.183849), -51.049417),
    "Hf": ((30.617033, 15.145351, 54.933548, 4.096253, 16.896157),
         (1.795613, 9.934469, 0.000739, 76.189707, 0.175914), -49.719838),
    "Ta": ((31.066358, 15.341823, 49.278296, 4.577665, 16.828321),
         (1.708732, 9.618455, 0.000760, 66.346202, 0.168002), -44.119025),
    "W": ((31.507901, 15.682498, 37.960127, 4.885509, 16.792113),
        (1.629485, 9.446448, 0.000898, 59.980675, 0.160798), -32.864576),
    "Re": ((31.888456, 16.117103, 42.390296, 5.211669, 16.767591),
         (1.549238, 9.233474, 0.000689, 54.516371, 0.152815), -37.412681),
    "Os": ((32.210298, 16.678440, 48.559907, 5.455839, 16.735532),
         (1.473531, 9.049695, 0.000519, 50.210201, 0.145771), -43.677954),
    "Ir": ((32.004437, 1.975454, 17.070104, 15.939454, 5.990003),
         (1.353767, 81.014172, 0.128093, 7.661196, 26.659403), 4.018893),
    "Pt": ((31.273891, 18.445441, 17.063745, 5.555933, 1.575270),
         (1.316992, 8.797154, 0.124741, 40.177994, 1.316997), 4.050394),
    "Au": ((16.777389, 19.317156, 32.979682, 5.595453, 10.576854),
         (0.122737, 8.621570, 1.256902, 38.008821, 0.000601), -6.279078),
    "Hg": ((16.839889, 20.023823, 28.428565, 5.881564, 4.714706),
         (0.115905, 8.256927, 1.195250, 39.247226, 1.195250), 4.076478),
    "Tl": ((16.630795, 19.386615, 32.808570, 1.747191, 6.356862),
         (0.110704, 7.181401, 1.119730, 90.660262, 26.014978), 4.066939),
    "Pb": ((16.419567, 32.738592, 6.530247, 2.342742, 19.916475),
         (0.105499, 1.055049, 25.025890, 80.906596, 6.664449), 4.049824),
    "Bi": ((16.282274, 32.725137, 6.678302, 2.694750, 20.576559),
         (0.101180, 1.002287, 25.714145, 77.057550, 6.291882), 4.040914),
    "Po": ((16.289164, 32.807170, 21.095164, 2.505901, 7.254589),
         (0.098121, 0.966265, 6.046622, 76.598071, 28.096128), 4.046556),
    "At": ((16.011461, 32.615549, 8.113899, 2.884082, 21.377867),
         (0.092639, 0.904416, 26.543256, 68.372961, 5.499512), 3.995684),
    "Rn": ((16.070228, 32.641105, 21.489659, 2.299218, 9.480184),
         (0.090437, 0.876409, 5.239687, 69.188477, 27.632640), 4.020977),
    "Fr": ((16.007386, 32.663830, 21.594351, 1.598497, 11.121192),
         (0.087031, 0.840187, 4.954467, 199.805805, 26.905106), 4.003472),
    "Ra": ((32.563691, 21.396671, 11.298093, 2.834688, 15.914965),
         (0.801980, 4.590666, 22.758973, 160.404392, 0.083544), 3.981773),
    "Ac": ((15.914053, 32.535042, 21.553976, 11.433394, 3.612409),
         (0.080511, 0.770669, 4.352206, 21.381622, 130.500748), 3.939212),
    "Th": ((15.784024, 32.454898, 21.849222, 4.239077, 11.736191),
         (0.077067, 0.735137, 4.097976, 109.464113, 20.512138), 3.922533),
    "Pa": ((32.740208, 21.973674, 12.957398, 3.683832, 15.744058),
         (0.709545, 4.050881, 19.231542, 117.255006, 0.074040), 3.886066),
    "U": ((15.679275, 32.824305, 13.660459, 3.687261, 22.279435),
        (0.071206, 0.681177, 18.236157, 112.500040, 3.930325), 3.854444),
    "Np": ((32.999899, 22.638076, 14.219973, 3.672950, 15.683245),
         (0.657086, 3.854918, 17.435474, 109.464485, 0.068033), 3.769391),
    "Pu": ((33.281176, 23.148545, 15.153755, 3.031492, 15.704215),
         (0.634999, 3.856168, 16.849736, 121.292040, 0.064857), 3.664200),
    "Am": ((33.435163, 23.657259, 15.576339, 3.027023, 15.746100),
         (0.612785, 3.792942, 16.195778, 117.757005, 0.061755), 3.541160),
    "Cm": ((15.804837, 33.480800, 24.150198, 3.655563, 15.499866),
         (0.058619, 0.590160, 3.674720, 100.736192, 15.408296), 3.390840),
    "Bk": ((15.889072, 33.625285, 24.710380, 3.707139, 15.839268),
         (0.055503, 0.569571, 3.615472, 97.694787, 14.754303), 3.213169),
    "Cf": ((33.794074, 25.467693, 16.048486, 3.657525, 16.008982),
         (0.550447, 3.581973, 14.357388, 96.064975, 0.052450), 3.005326),
}


def xray_form_factor(symbol, q):
    """X-ray atomic form factor f0(q) (electrons), Waasmaier-Kirfel 1995.

    Parameters
    ----------
    symbol : str
        Element symbol.
    q : float or array_like
        Scattering vector magnitude in inverse Angstrom
        (q = 4*pi*sin(theta)/lambda).

    Returns
    -------
    ndarray
        f0 evaluated at each q. f0(0) == Z.
    """
    try:
        a, b, c = _WK95_XRAY[symbol]
    except KeyError:
        raise KeyError(
            f"No X-ray form-factor coefficients tabulated for element "
            f"{symbol!r} (Waasmaier-Kirfel 1995 covers H..Cf)."
        ) from None
    q = np.asarray(q, dtype=float)
    s2 = (q / (4.0 * np.pi)) ** 2
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[:, None]
    flat = (a * np.exp(-b * s2.ravel()[None, :])).sum(axis=0) + c
    return flat.reshape(q.shape)


def _form_factors(unique_symbols, weighting):
    """Return ``{symbol: f(q)}`` where each ``f`` maps a q-array to an array.

    * ``"xray"``       -- q-dependent Waasmaier-Kirfel f0(q); f0(0) = Z.
    * ``"neutron"``    -- coherent scattering length b (q-independent).
    * ``"unweighted"`` -- 1.

    Returning callables lets both S(q) methods weight each q correctly:
    x-ray form factors fall off with q at element-specific rates (at
    q = 2.45 A^-1 f/Z is 0.81 for Ga but 0.71 for O), so constant-Z
    weighting mis-weights the partials at finite q.
    """
    if weighting == "xray":
        for s in unique_symbols:
            if s not in _WK95_XRAY:
                raise KeyError(
                    f"No X-ray form-factor coefficients tabulated for "
                    f"element {s!r} (Waasmaier-Kirfel 1995 covers H..Cf)."
                )
        return {s: (lambda q, _s=s: xray_form_factor(_s, q))
                for s in unique_symbols}
    if weighting == "neutron":
        try:
            b = {s: _NEUTRON_B[s] for s in unique_symbols}
        except KeyError as exc:
            raise KeyError(
                f"No neutron scattering length tabulated for element {exc}. "
                f"Add it to amorphgen.analysis.rdf._NEUTRON_B or use "
                f"weighting='xray' instead."
            ) from None
        return {s: (lambda q, _b=b[s]: np.full(np.shape(q), _b, dtype=float))
                for s in unique_symbols}
    if weighting == "unweighted":
        return {s: (lambda q: np.ones(np.shape(q), dtype=float))
                for s in unique_symbols}
    raise ValueError(
        f"weighting must be 'unweighted', 'xray', or 'neutron'; "
        f"got {weighting!r}"
    )



def _smooth_sq_weighted(q, s_q, n_per_bin, sigma_q):
    """Gaussian re-binning of a shell-averaged S(q), weighted by the number
    of q-vectors in each shell.

    Each output point is the mean of every q-vector within the kernel, so
    shells holding more vectors count more and empty (NaN) shells count
    nothing. Statistically this is just a wider, softer bin -- it reduces
    the 1/sqrt(N) speckle noise of the direct method without moving peaks.
    Keep ``sigma_q`` well below the FSDP width (~0.3 A^-1) to avoid damping
    it; 0.05-0.10 A^-1 is a sensible presentation range.
    """
    q = np.asarray(q, dtype=float)
    s = np.asarray(s_q, dtype=float)
    n = np.asarray(n_per_bin, dtype=float)
    valid = np.isfinite(s) & (n > 0)
    out = np.full_like(s, np.nan)
    if sigma_q <= 0 or not valid.any():
        return s.copy()
    qv, sv, nv = q[valid], s[valid], n[valid]
    for i, qi in enumerate(q):
        w = nv * np.exp(-0.5 * ((qv - qi) / sigma_q) ** 2)
        wsum = w.sum()
        if wsum > 0 and valid[i]:
            out[i] = (w * sv).sum() / wsum
    return out

def compute_structure_factor_direct(atoms_list, qmax=15.0, nq=300,
                                    weighting="xray", q_batch=4096,
                                    sigma_q=0.0, partials=False):
    """Compute S(q) directly from atomic positions via the Debye formula
    evaluated at reciprocal-lattice q-vectors.

    Avoids the rmax truncation that damps the first sharp diffraction
    peak (FSDP) in the FFT-of-g(r) approach. Q-vector enumeration uses
    the reciprocal lattice of each structure, so the q-resolution is
    limited only by the simulation cell size (q_min ~ 2*pi/L).

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Ensemble of structures (all assumed to have the same composition).
    qmax : float
        Maximum q in inverse-Angstrom.
    nq : int
        Number of q-bins between 0 and qmax for spherical averaging.
    weighting : {"xray", "neutron", "unweighted"}
        Per-element scattering factors used in the sum.
    q_batch : int
        Number of q-vectors processed per matmul. Tune for memory.
    sigma_q : float
        Optional Gaussian re-binning width in inverse Angstrom (default 0 =
        off). Averages the shell values with weights ``n_per_bin`` to reduce
        the 1/sqrt(N) speckle noise; keep it well below the FSDP width
        (~0.3 A^-1). The unsmoothed values are returned as ``s_q_raw``.
    partials : bool
        Also return the Faber-Ziman partial structure factors S_ab(q) for
        every element pair, computed from the same reciprocal-lattice sum
        (so they resolve the FSDP like the total). Per q-vector,
        ``S_ab = 1 + (Re<F_a F_b*>/sqrt(N_a N_b) - delta_ab)/sqrt(c_a c_b)``
        with ``F_a = sum_{i in a} exp(i q.r_i)``; the weighted sum
        ``sum_ab (2-delta_ab) c_a c_b f_a f_b S_ab / <f>^2`` reproduces
        ``s_q`` exactly. Partials are pure geometry (independent of
        ``weighting``) and are smoothed with ``sigma_q`` like the total.

    Returns
    -------
    dict
        ``{"q": list[float], "s_q": list[float], "n_per_bin": list[int]}``
        (plus ``"s_q_raw"`` when ``sigma_q > 0``, and ``"partials"``:
        ``{"A-B": list[float], ...}`` when ``partials=True``).
        ``n_per_bin`` is the number of reciprocal-lattice vectors in each
        spherical shell — small values (1-3) indicate noisy estimates at
        low q.
    """
    atoms0 = atoms_list[0]
    syms0 = atoms0.get_chemical_symbols()
    unique = sorted(set(syms0))
    ff = _form_factors(unique, weighting)      # {symbol: f(q) callable}
    n_atoms = len(atoms0)

    # Composition fractions for the Faber-Ziman normalisation. The scattering
    # factors are q-dependent (x-ray f0(q)), so <f>, <f^2> and the
    # self-scattering offset are evaluated per q-vector inside the batch
    # loop rather than once here.
    c = np.array([syms0.count(s) for s in unique], dtype=float) / n_atoms

    # Spherical-shell accumulators
    q_edges = np.linspace(0.0, qmax, nq + 1)
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    sq_sum = np.zeros(nq)
    sq_count = np.zeros(nq, dtype=int)
    pair_keys = [(a, b) for i, a in enumerate(unique) for b in unique[i:]]
    part_sum = {k: np.zeros(nq) for k in pair_keys} if partials else None

    for atoms in atoms_list:
        positions = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        # Map each atom to its element's column in the per-q form-factor
        # matrix built inside the batch loop.
        elem_idx = np.array([unique.index(s) for s in syms])
        N = len(atoms)
        n_species = np.array([syms.count(s) for s in unique], dtype=float)

        # Reciprocal lattice: rows are b_i with a_i . b_j = 2pi delta_ij
        recip = 2.0 * np.pi * np.linalg.inv(atoms.cell.array).T  # (3,3)
        recip_min = float(np.linalg.norm(recip, axis=1).min())
        if recip_min == 0:
            continue
        n_max = int(np.ceil(qmax / recip_min)) + 1

        # Build all integer triplets (n1,n2,n3), exclude origin, build q
        ns = np.arange(-n_max, n_max + 1)
        grid = np.array(np.meshgrid(ns, ns, ns, indexing="ij")).reshape(3, -1).T
        # drop origin
        grid = grid[~np.all(grid == 0, axis=1)]
        q_vecs = grid @ recip                          # (M, 3)
        q_mags = np.linalg.norm(q_vecs, axis=1)
        keep = q_mags <= qmax
        q_vecs = q_vecs[keep]
        q_mags = q_mags[keep]

        # Process in batches to control memory
        M = q_vecs.shape[0]
        for start in range(0, M, q_batch):
            stop = min(start + q_batch, M)
            qb = q_vecs[start:stop]                    # (m, 3)
            qm = q_mags[start:stop]                    # (m,)
            phases = qb @ positions.T                  # (m, N)

            # Per-q-vector form factors: F[k, e] = f_e(|q_k|); one column per
            # element, then broadcast to every atom of that element.
            F = np.stack([ff[s](qm) for s in unique], axis=1)   # (m, n_elem)
            f_atoms = F[:, elem_idx]                             # (m, N)
            f_mean = F @ c                                       # (m,)
            f2_mean = (F * F) @ c                                # (m,)
            # Self-scattering offset: raw |Σ f e^{iqr}|²/(N<f>²) tends to
            # <f²>/<f>² at high q (uncorrelated phases), not 1. Subtracting
            # it gives the Faber-Ziman total S(q) with S(q→∞) = 1. Zero for
            # unweighted/monatomic (f² = f = 1). q-dependent for x-rays.
            safe = np.where(f_mean == 0, 1.0, f_mean)
            self_offset = f2_mean / (safe * safe) - 1.0
            denom = N * safe * safe

            # |Σ f_i e^{i q·r_i}|² = (Σ f cos)² + (Σ f sin)²
            cos_ph = np.cos(phases)
            sin_ph = np.sin(phases)
            cos_sum = (f_atoms * cos_ph).sum(axis=1)
            sin_sum = (f_atoms * sin_ph).sum(axis=1)
            sq_vals = ((cos_sum * cos_sum + sin_sum * sin_sum) / denom
                       - self_offset)

            # Bin by |q|
            bin_idx = np.clip(((q_mags[start:stop] / qmax) * nq).astype(int),
                              0, nq - 1)
            np.add.at(sq_sum, bin_idx, sq_vals)
            np.add.at(sq_count, bin_idx, 1)

            if partials:
                # Per-species amplitudes F_a = sum_{i in a} e^{i q.r_i}
                C = {s: cos_ph[:, elem_idx == k].sum(axis=1)
                     for k, s in enumerate(unique)}
                S = {s: sin_ph[:, elem_idx == k].sum(axis=1)
                     for k, s in enumerate(unique)}
                for (a, b) in pair_keys:
                    ka, kb = unique.index(a), unique.index(b)
                    na, nb = n_species[ka], n_species[kb]
                    ca, cb = na / N, nb / N
                    # Ashcroft-Langreth partial, then Faber-Ziman
                    al = (C[a] * C[b] + S[a] * S[b]) / np.sqrt(na * nb)
                    fz = 1.0 + (al - (1.0 if a == b else 0.0)) / np.sqrt(ca * cb)
                    np.add.at(part_sum[(a, b)], bin_idx, fz)

    sq = np.where(sq_count > 0, sq_sum / np.maximum(sq_count, 1), np.nan)
    out = {
        "q": q_centres.tolist(),
        "s_q": sq.tolist(),
        "n_per_bin": sq_count.tolist(),
    }
    if sigma_q > 0:
        out["s_q_raw"] = sq.tolist()
        out["s_q"] = _smooth_sq_weighted(q_centres, sq, sq_count,
                                         sigma_q).tolist()
    if partials:
        out["partials"] = {}
        for (a, b), tot in part_sum.items():
            p = np.where(sq_count > 0, tot / np.maximum(sq_count, 1), np.nan)
            if sigma_q > 0:
                p = _smooth_sq_weighted(q_centres, p, sq_count, sigma_q)
            out["partials"][f"{a}-{b}"] = p.tolist()
    return out


def compute_structure_factor(atoms_list, pair=None, qmax=15.0, nq=300,
                             rmax=None, weighting="unweighted"):
    """Compute the structure factor S(q) from g(r) via Fourier transform.

    Parameters
    ----------
    atoms_list : list[ase.Atoms]
        Ensemble of structures.
    pair : str, optional
        Specific A-B partial (e.g. ``"Ga-O"``). If ``None``, computes the
        total S(q). The ``weighting`` argument only affects the total
        case; explicit partials are always returned as their own
        Faber-Ziman S_AB(q).
    qmax : float
        Maximum q in inverse-Angstrom (default 15.0).
    nq : int
        Number of q points (default 300).
    rmax : float, optional
        Max radius for the underlying g(r). ``None`` = auto (half cell).
    weighting : {"unweighted", "xray", "neutron"}, default ``"unweighted"``
        How to combine partials into the total S(q):

        * ``"unweighted"`` — single FT of the all-atom g(r). Fast,
          useful for ensemble-vs-ensemble comparisons. Does NOT match
          experimental X-ray/neutron S(Q) in general because it omits
          per-element scattering weights.
        * ``"xray"`` — Faber-Ziman partials weighted by atomic numbers
          squared (Z_A·Z_B). The Z² approximation is exact only at
          q = 0; quantitatively good below q ~ 5 inverse-Angstrom
          (covers the FSDP and main-peak region).
        * ``"neutron"`` — same Faber-Ziman combination but weighted by
          tabulated neutron coherent scattering lengths (b_A·b_B). Uses
          a built-in table of ~50 common elements; raises KeyError for
          unsupported species.

    Returns
    -------
    dict
        ``{"q": list[float], "s_q": list[float]}``.

    Notes
    -----
    For X-ray S(Q) of amorphous oxides the FSDP at ~1.5-2.5
    inverse-Angstrom is dominated by heavy-atom cation-cation
    correlations and cancels in the unweighted sum because
    the cation-anion partial dips at the same q. The ``"xray"``
    weighting recovers it. See ``examples/test_structure_factor.py``
    for a worked example on a-Ga2O3 (Kaewmeechai et al., Phys. Rev. B
    111, 035203, 2025).
    """
    if weighting not in ("unweighted", "xray", "neutron"):
        raise ValueError(
            f"weighting must be 'unweighted', 'xray', or 'neutron'; "
            f"got {weighting!r}"
        )

    # ── Weighted-total path (Faber-Ziman combination of partials) ────────
    if pair is None and weighting in ("xray", "neutron"):
        atoms = atoms_list[0]
        syms = atoms.get_chemical_symbols()
        unique = sorted(set(syms))
        if not unique:
            return {"q": [], "s_q": []}
        total_n = len(syms)
        fractions = {s: syms.count(s) / total_n for s in unique}
        # q-dependent scattering factors (x-ray f0(q); constant b for
        # neutrons) -- the Faber-Ziman weights are evaluated per q below.
        ff = _form_factors(unique, weighting)

        # Compute every unique partial once
        partials = {}
        q_arr = None
        for i, s1 in enumerate(unique):
            for s2 in unique[i:]:
                pstr = f"{s1}-{s2}"
                sq = compute_structure_factor(
                    atoms_list, pair=pstr, qmax=qmax, nq=nq, rmax=rmax,
                    weighting="unweighted",   # avoid recursion
                )
                if q_arr is None:
                    q_arr = np.array(sq["q"])
                partials[(s1, s2)] = np.array(sq["s_q"])

        if q_arr is None:
            return {"q": [], "s_q": []}

        # Faber-Ziman: w_AB(q) = (2 - delta_AB) c_A c_B f_A(q) f_B(q) / <f(q)>^2
        # Evaluated per q because x-ray form factors fall off with q at
        # element-specific rates (constant-Z weighting mis-weights the
        # partials at finite q; see _form_factors).
        f = {s: ff[s](q_arr) for s in unique}                     # arrays
        mean_f = sum(fractions[s] * f[s] for s in unique)
        norm = np.where(mean_f == 0, 1.0, mean_f * mean_f)

        s_total = np.zeros_like(q_arr)
        weight_sum = np.zeros_like(q_arr)
        for (s1, s2), s_part in partials.items():
            mult = 1.0 if s1 == s2 else 2.0
            w = mult * fractions[s1] * fractions[s2] * f[s1] * f[s2] / norm
            s_total += w * s_part
            weight_sum += w
        # sum_AB w_AB == 1 identically; the division only guards degenerate f.
        s_total = np.where(weight_sum > 0, s_total / np.where(weight_sum > 0, weight_sum, 1.0), s_total)

        return {"q": q_arr.tolist(), "s_q": s_total.tolist()}

    # ── Unweighted / partial path (original behaviour) ───────────────────
    # Always transform the RAW g(r): smearing is a presentation choice for
    # RDF plots and would damp S(q) by exp(-q^2 sigma^2 / 2) (~16% at q=12).
    rdf_data = compute_rdf(atoms_list, pair=pair, rmax=rmax, nbins=500,
                           sigma=0.0)
    r = np.array(rdf_data["r"])
    g_r = np.array(rdf_data["g_r"])
    dr = r[1] - r[0] if len(r) > 1 else 0.04

    atoms = atoms_list[0]
    n = len(atoms)
    vol = atoms.get_volume()

    if pair is not None:
        p1, p2 = pair.split("-")
        syms = atoms.get_chemical_symbols()
        if sum(1 for s in syms if s == p2) == 0:
            return {"q": [], "s_q": []}

    # Faber-Ziman: the transform prefactor is the TOTAL number density rho0
    # for partials as well as the total (g_ab itself is already normalised
    # per-species, so g_ab -> 1). Using the partner density n_b/V here scales
    # every partial by c_b and breaks the FZ combination -- with equal
    # scattering lengths the weighted total must equal the unweighted total,
    # and the partner-density form gives exactly half for a 50/50 binary.
    # (n - 1) matches the unweighted-total path so that identity is exact.
    rho = (n - 1) / vol

    if rho <= 0 or len(r) == 0:
        return {"q": [], "s_q": []}

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    q_values = np.linspace(0.1, qmax, nq)
    s_q = np.ones(nq)

    for iq, q in enumerate(q_values):
        qr = q * r
        sinc_qr = np.where(qr > 1e-12, np.sin(qr) / qr, 1.0)
        # 3D isotropic transform: S(q) = 1 + 4*pi*rho * INT r^2 (g-1) sinc(qr) dr.
        # The r^2 (not r) makes rho*INT dimensionless (L^-3 * L^3); with only r
        # the integrand is short one factor of r and S(q) is systematically
        # wrong. The xray/neutron Faber-Ziman path recurses through here, so it
        # inherits this transform.
        integrand = r ** 2 * (g_r - 1.0) * sinc_qr
        s_q[iq] = 1.0 + 4.0 * np.pi * rho * _trapz(integrand, dx=dr)

    return {"q": q_values.tolist(), "s_q": s_q.tolist()}


def compute_averaged_rdf(atoms_list, pair=None, rmax=None, nbins=200):
    """Compute RDF per structure with mean and std."""
    if rmax is None:
        half_cells = [min(a.cell.lengths()) / 2 for a in atoms_list]
        rmax = float(np.floor(min(half_cells) * 10) / 10)

    dr = rmax / nbins
    r_centres = np.linspace(dr / 2, rmax - dr / 2, nbins)
    shell_vols = 4 * np.pi * r_centres**2 * dr

    all_g_r = []

    for atoms in atoms_list:
        idx_i, idx_j, dists = neighbor_list('ijd', atoms, cutoff=rmax)
        syms = np.array(atoms.get_chemical_symbols())
        vol = atoms.get_volume()
        n = len(atoms)

        if pair is not None:
            p1, p2 = pair.split("-")
            n_source = int(np.sum(syms == p1))
            n_target = int(np.sum(syms == p2))
            if p1 == p2:
                rho_target = (n_target - 1) / vol
            else:
                rho_target = n_target / vol
            mask = (syms[idx_i] == p1) & (syms[idx_j] == p2)
            pair_dists = dists[mask]
        else:
            n_source = n
            rho_target = (n - 1) / vol
            pair_dists = dists

        if len(pair_dists) == 0 or n_source == 0 or rho_target <= 0:
            all_g_r.append(np.zeros(nbins))
            continue

        in_range = (pair_dists > 0) & (pair_dists < rmax)
        if not np.any(in_range):
            all_g_r.append(np.zeros(nbins))
            continue
        hist, _ = np.histogram(pair_dists[in_range], bins=nbins,
                               range=(0, rmax))

        g_r = np.zeros(nbins)
        valid = shell_vols > 0
        g_r[valid] = hist[valid] / (n_source * rho_target * shell_vols[valid])
        all_g_r.append(g_r)

    all_g_r = np.array(all_g_r)

    return {
        "r": r_centres.tolist(),
        "g_r_mean": np.mean(all_g_r, axis=0).tolist(),
        "g_r_std": np.std(all_g_r, axis=0).tolist(),
        "n_structures": len(atoms_list),
    }
