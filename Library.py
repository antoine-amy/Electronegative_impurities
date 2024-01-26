# Diffusion constant given in cm^2/s
# Solubility given as fraction 
# Activation energy given in eV 
# Mass given in gram 
# Volume given in liter
# Surface area given in cm^2
# Thickness given in cm 
# Molar mass  given in g/mol

Material = {
    'Teflon':{  # http://arxiv.org/abs/1703.09144
        'Nitrogen':{'Diffusion Constant': 15.1E-8, 'Solubility': 0.107, 'Activation Energy': 0.17},
        'Oxygen':{'Diffusion Constant': 31.4E-8, 'Solubility': 0.22, 'Activation Energy': 0.17}, 
        'Krypton':{'Diffusion Constant': 5.6E-8, 'Solubility': 0.58, 'Activation Energy': 0.17},
        'Xenon':{'Diffusion Constant': 0.8E-8, 'Solubility': 0.89, 'Activation Energy': 0.17},
        'Argon':{'Diffusion Constant': 16.8E-8, 'Solubility': 0.088, 'Activation Energy': 0.17},
        'Helium':{'Diffusion Constant': 1270E-8, 'Solubility': 0.033, 'Activation Energy': 0.17}
    },
    'Teflon Yale':{  # http://arxiv.org/abs/1703.09144
        'Oxygen':{'Diffusion Constant': 31.4E-8, 'Solubility': 0.22, 'Activation Energy': 0.47}, 
    },
    'Viton':{  # http://arxiv.org/abs/1703.09144
        'Nitrogen':{'Diffusion Constant': 2.2E-8, 'Solubility': 0.51, 'Activation Energy': 0.17},
        'Oxygen':{'Diffusion Constant': 6.8E-8, 'Solubility': 0.22, 'Activation Energy': 0.17},
        'Krypton':{'Diffusion Constant': 1.25E-8, 'Solubility': 0.23, 'Activation Energy': 0.17}, 
        'Xenon':{'Diffusion Constant': 1.7E-8, 'Solubility': 0.15, 'Activation Energy': 0.17},
        'Argon':{'Diffusion Constant': 4.0E-8, 'Solubility': 0.083, 'Activation Energy': 0.17},
        'Helium':{'Diffusion Constant': 436E-8, 'Solubility': 0.093, 'Activation Energy': 0.17}
    },
    'PE1':{  # http://arxiv.org/abs/1703.09144
        'Nitrogen':{'Diffusion Constant': 16E-8, 'Solubility': 0.021, 'Activation Energy': 0.17},
        'Oxygen':{'Diffusion Constant': 39E-8, 'Solubility': 0.018, 'Activation Energy': 0.17},
        'Krypton':{'Diffusion Constant': 6.4E-8, 'Solubility': 0.093, 'Activation Energy': 0.17},
        'Xenon':{'Diffusion Constant': 2.0E-8, 'Solubility': 0.55, 'Activation Energy': 0.17},
        'Argon':{'Diffusion Constant': 20E-8, 'Solubility': 0.047, 'Activation Energy': 0.17},
        'Helium':{'Diffusion Constant': 435E-8, 'Solubility': 0.0064, 'Activation Energy': 0.17}
    },
    'Kapton':{
        'Oxygen':{'Diffusion Constant': 31.4E-8, 'Solubility': 0.22, 'Activation Energy': 0.17}
    },  
    'Kapton Yale':{
        'Oxygen':{'Diffusion Constant': 31.4E-8, 'Solubility': 0.22, 'Activation Energy': 0.47}
    }
}

System = {
    'EXO-200':{
        'Xenon Mass': 200000,
        'Teflon':{
                'EXO-Teflon': {'Volume': 0.693, 'Area': 9200.0, 'Thickness': 0.15},
                'EXO-Acrylic': {'Volume': 0.693, 'Area': 9200.0, 'Thickness': 2.0}
        },
        'Teflon Yale':{
                'EXO-Teflon': {'Volume': 0.693, 'Area': 9200.0, 'Thickness': 0.15},
                'EXO-Acrylic': {'Volume': 0.693, 'Area': 9200.0, 'Thickness': 2.0}
        }
    },
    'nEXO':{
        'Xenon Mass': 5000000,
        'Kapton':{
            'nEXO-Kapton':{'Volume': 0.0625, 'Area': 12500.0, 'Thickness': 0.005}
        },
        'Kapton Yale':{
            'nEXO-Kapton':{'Volume': 0.0625, 'Area': 12500.0, 'Thickness': 0.005}
        }
    },
    'LZ':{
        'Xenon Mass': 10000000,
        'Material':{
            'Teflon':{'Volume': 143.0, 'Area': 4600.0, 'Thickness': 2.0}
        }
    },
    'YLXPS':{
        'Xenon Mass': 2170,
        'Teflon':{
                'EXO-Teflon': {'Volume': 5.3E-3, 'Area': pow(5.5*2.54, 2)*2, 'Thickness': 0.15},
                'Stock-Teflon': {'Volume': 0.01639, 'Area': pow(4*2.54, 2)*2, 'Thickness': 0.635},
                'Stock-Teflon Thick': {'Volume': 0.093*1, 'Area': 45.6*2 + 48.64*1, 'Thickness': 2.03},
                'Columbia-Teflon': {'Volume': 285.08, 'Area': (17.8*15.4)*2, 'Thickness': 1.04}
        }
    },
    'Columbia':{
        'Xenon Mass': 2170,
        'Material':{
            'Teflon':{'Volume': 285.08, 'Area': 274.12, 'Thickness': 1.04}
        }
    }
}

Gas = {
    'Oxygen':{'Abundance in Air': 0.21, 'Molar Mass': 32},
    'Nitrogen':{'Abundance in Air': 0.78,'Molar Mass': 28},
    'Krypton':{'Abundance in Air': 0.009, 'Molar Mass': 84},
    'Xenon':{'Abundance in Air': 11.5E-6, 'Molar Mass': 136},
    'Argon':{'Abundance in Air': 0.0093, 'Molar Mass': 40},
    'Helium':{'Abundance in Air': 5E-6, 'Molar Mass': 4},
}