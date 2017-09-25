import optoanalysis as oa

data = oa.load_data("testData.raw")

OmegaTrap, A, Damping,  _, _ = data.get_fit_auto(70e3)

OmegaTrap = OmegaTrap.n
A = A.n
Damping = Damping.n

Ftrap = OmegaTrap/(2*oa.pi)

# next 3 lines are just me getting the pressure value from a file
with open("testDataPressure.dat", 'r') as file:
    for line in file:
        pressure = float(line.split("mbar")[0])

pmbar = pressure # can get from file if pressure.log exists
        
radius_rashid, M_rashid, ConvFactor_rashid = data.extract_parameters(pmbar, 0.15, method="rashid")

radius_chang, M_chang, ConvFactor_chang = data.extract_parameters(pmbar, 0.15, method="chang")
N = 3

z, t, _, _  = data.filter_data(Ftrap, FractionOfSampleFreq=N, MakeFig=False, ShowFig=False)
SampleFreq = data.SampleFreq/N

radius_potentials, radius_potentials_err, _, _ = oa.fit_radius_from_potentials(z, SampleFreq, Damping)

z2, t2, _, _  = data.filter_data(2*Ftrap, FractionOfSampleFreq=3, MakeFig=False, ShowFig=False)

z0, ConvFactor = oa.calc_z0_and_conv_factor_from_ratio_of_harmonics(z, z2, NA=0.999)

mass_equipartition_and_harmonic_comparision = oa.calc_mass_from_z0(z0, OmegaTrap)

radius_equipartition_and_harmonic_comparision = oa.calc_radius_from_mass(mass_equipartition_and_harmonic_comparision)

mass_using_convfactor_from_harmonic_comparision = oa.calc_mass_from_fit_and_conv_factor(A, Damping, ConvFactor)

radius_using_convfactor_from_harmonic_comparision = oa.calc_radius_from_mass(mass_using_convfactor_from_harmonic_comparision)


print(radius_rashid*1e9)
print(radius_chang*1e9)
print(radius_potentials*1e9)
print(radius_equipartition_and_harmonic_comparision*1e9)
print(radius_using_convfactor_from_harmonic_comparision*1e9)
