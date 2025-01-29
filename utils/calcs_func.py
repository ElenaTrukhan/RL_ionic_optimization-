from ase.calculators.eam import EAM
EAM_Al = EAM(potential = "EAM/Al.eam.alloy")
EAM_AlFe = EAM(potential = "EAM/AlFe.fs")
EAM_Fe = EAM(potential = "EAM/Fe.eam.fs")


def func_for_calc(struct): 
    if len(set(struct.species)) == 2: 
        return EAM_AlFe
    else: 
        if struct.species[0].symbol == 'Al': 
            return EAM_Al
        else: 
            return EAM_Fe
