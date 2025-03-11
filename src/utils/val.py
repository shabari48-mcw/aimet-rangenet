from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_preparer import prepare_model
import torch


def validate_model(model,dummy_input)->None :
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print("\n\nPerforming Model Validation.........\n\n")
    
    if ModelValidator.validate_model(model,dummy_input) == True:
        print(f" ✅ Model Validation Passed")
        return model
    else:
        model=prepare_model(model)
        
        if ModelValidator.validate_model(model,dummy_input) == True:
            print(f"✅ Model Validation Passed\n\n")
            return model
        else :
            print(f"❌ Model Validation Failed\n\n")
            raise ValueError("Error Validating Model")
        
