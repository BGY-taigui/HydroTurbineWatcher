import json
import os
import numpy as np

def json_feature_reader(file_name: str):

    try:
        path = os.path.normpath(file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"success {path} (type: {type(data).__name__})")
        return data

    except FileNotFoundError as e:
        print(f"error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def json_feature_writer(filename: str, feature_data: dict):

    try:
        path = os.path.normpath(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(feature_data, f, ensure_ascii=False, indent=4)
        print(f"success {path} (type: {type(feature_data).__name__})")
    except Exception as e:
        print(f"Unexpected error: {e}")

def json_feature_blender_1D(feature_data_1: dict, feature_data_2: dict, blending_ratio: float):

    blended_feature_data = feature_data_1

    if len(feature_data_1["Feature Manipulation"]) !=1 and len(feature_data_2["Feature Manipulation"]) != 1:
        raise ValueError("Feature Manipulation length is not 1 in one of the feature data.")

    blended_feature_data["Feature Manipulation"][0]["mean values"] = (np.array(feature_data_1["Feature Manipulation"][0]["mean values"])* blending_ratio + np.array(feature_data_2["Feature Manipulation"][0]["mean values"])* (1 - blending_ratio)).tolist()

    blended_feature_data["Feature Manipulation"][0]["standard deviation values"] = (np.array(feature_data_1["Feature Manipulation"][0]["standard deviation values"])* blending_ratio + np.array(feature_data_2["Feature Manipulation"][0]["standard deviation values"])* (1 - blending_ratio)).tolist()

    blended_feature_data["Anomaly Detection"]["mean values"] = (np.array(feature_data_1["Anomaly Detection"]["mean values"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["mean values"]) * (1 - blending_ratio)).tolist()


    # Blending eigen vectors
    
    eigenvectors_1 =  np.array(feature_data_1["Anomaly Detection"]["eigenvectors"]).T
    eigenvectors_2 = np.array(feature_data_2["Anomaly Detection"]["eigenvectors"]).T

    eigenvalues_1 = feature_data_1["Anomaly Detection"]["variance"]
    eigenvalues_2 = feature_data_2["Anomaly Detection"]["variance"]
    eighenvector_length = len(eigenvectors_1)

    corresponding_indexes = []
    corresponding_indexes_rem = [item for item in range(eighenvector_length)]

    #TODO ここのアルゴリズム合ってるか要確認
    for i in range(eighenvector_length):
        diff = [abs(np.dot(eigenvectors_1[i],eigenvectors_2[j])) for j in corresponding_indexes_rem]
        min_index = diff.index(min(diff))
        corresponding_indexes.append(corresponding_indexes_rem[min_index])
        corresponding_indexes_rem.remove(corresponding_indexes_rem[min_index])
    
    sorted_eigenvectors_1 = eigenvectors_1
    sorted_eigenvalues_1 = eigenvalues_1
    sorted_eigenvectors_2 = np.array([eigenvectors_2[i] for i in corresponding_indexes])
    sorted_eigenvalues_2 = [eigenvalues_2[i] for i in corresponding_indexes]
    
    #TODO 転置の順番があっているか要確認
    P_1 = np.dot(sorted_eigenvectors_1.T , sorted_eigenvectors_1)
    P_2 = np.dot(sorted_eigenvectors_2.T , sorted_eigenvectors_2)

    P = P_1 * blending_ratio + P_2 * (1 - blending_ratio)

    P_eigenvalues,P_eigenvectors = np.linalg.eigh(P)

    blended_eigenvectors = P_eigenvectors[::-1][:eighenvector_length].T.tolist()

    #blended_eigenvectors = P_eigenvectors[:,eighenvector_length].T.tolist()
    #blended_eigenvectors = []
    #for ev1, ev2 in zip(eigenvectors_1, eigenvectors_2):
        #blended_eigenvector = (np.array(ev1) * blending_ratio + np.array(ev2) * (1 - blending_ratio)).tolist()
        #blended_eigenvectors.append(blended_eigenvector)

    blended_feature_data["Anomaly Detection"]["eigenvectors"] = blended_eigenvectors

    blended_feature_data["Anomaly Detection"]["variance"] = (np.array(sorted_eigenvalues_1) * blending_ratio + np.array(sorted_eigenvalues_2) * (1 - blending_ratio)).tolist()
    
    #(np.array(feature_data_1["Anomaly Detection"]["variance"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["variance"]) * (1 - blending_ratio)).tolist()

    blended_feature_data["Anomaly Detection"]["threshold for T2"] = (np.array(feature_data_1["Anomaly Detection"]["threshold for T2"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["threshold for T2"]) * (1 - blending_ratio)).tolist()

    blended_feature_data["Anomaly Detection"]["threshold for Q"] = (np.array(feature_data_1["Anomaly Detection"]["threshold for Q"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["threshold for Q"]) * (1 - blending_ratio)).tolist()

    return blended_feature_data

def feature_printer(feature_data: dict):

    print("Training Time:",feature_data["Training Time"])

    print("All Models:",feature_data["All Models"])

    print("A length of the feature data :",len(feature_data["Feature Manipulation"]))

    print("Normalization Settings:",feature_data["Feature Manipulation"][0]["normalization settings"])

    #print(feature_data["Feature Manipulation"][0]["mean values"])
    print("Feature Manipulation/mean values length",len(feature_data["Feature Manipulation"][0]["mean values"]))
    print("Feature Manipulation/mean values 0", feature_data["Feature Manipulation"][0]["mean values"][0])

    #print(feature_data["Feature Manipulation"][0]["standard deviation values"])
    print("Feature Manipulation/standard deviation values length",len(feature_data["Feature Manipulation"][0]["standard deviation values"]))
    print("Feature Manipulation/standard deviation values 0", feature_data["Feature Manipulation"][0]["standard deviation values"][0])

    print("Feature Manipulation/min values:",feature_data["Feature Manipulation"][0]["min values"])

    print("Feature Manipulation/max values:",feature_data["Feature Manipulation"][0]["max values"])

    print("Running Mode of Anomaly Detection:",feature_data["Running Mode of Anomaly Detection"])

    #print(feature_data["Anomaly Detection"]["mean values"])
    print("Anomaly Detection/mean values length",len(feature_data["Anomaly Detection"]["mean values"]))
    print("Anomaly Detection/mean values 0", feature_data["Anomaly Detection"]["mean values"][0])

    #print(feature_data["Anomaly Detection"]["eigenvectors"])
    print("Anomaly Detection/eigenvectors length",len(feature_data["Anomaly Detection"]["eigenvectors"]))
    print("Anomaly Detection/eigenvectors 0", feature_data["Anomaly Detection"]["eigenvectors"][0])

    #print(feature_data["Anomaly Detection"]["variance"])
    print("Anomaly Detection/variance length",len(feature_data["Anomaly Detection"]["variance"]))
    print("Anomaly Detection/variance 0", feature_data["Anomaly Detection"]["variance"][0])
    print("Anomaly Detection/variance:", feature_data["Anomaly Detection"]["variance"])

    print("Anomaly Detection/threshold for T2:", feature_data["Anomaly Detection"]["threshold for T2"])

    print("Anomaly Detection/threshold for Q:", feature_data["Anomaly Detection"]["threshold for Q"])

    print("Anomaly Detection (Batch):", feature_data["Anomaly Detection (Batch)"])

    print("Clustering:", feature_data["Clustering"])

    print("Classification:", feature_data["Classification"])

    print("Custom Info:", feature_data["Custom Info"])

class Model_Case:
    def __init__(
            self,case_one_modelfile: str,
            case_two_modelfile: str,
            output_modelfile: str,
            blending_ratio: float):
        self.case_one_modelfile = case_one_modelfile
        self.case_two_modelfile = case_two_modelfile
        self.output_modelfile = output_modelfile
        self.blending_ratio = blending_ratio

    def get_case_one_modelfile(self):
        return self.case_one_modelfile
    
    def get_case_two_modelfile(self):
        return self.case_two_modelfile

    def get_output_modelfile(self):
        return self.output_modelfile
    
    def get_blending_ratio(self):
        return self.blending_ratio

def get_swirl_num(n11,q11):
    d1 = 0.1961
    d2 = 0.1578

    Sw = np.pi **2 /480 * d2**2/d1**2 * n11 * (1/q11 - 1/(0.0067*n11))

    return Sw

def get_blending_ratio(n11_1, q11_1, n11_2, q11_2, n11_target, q11_target):
    Sw1 = get_swirl_num(n11_1, q11_1)
    Sw2 = get_swirl_num(n11_2, q11_2)
    Sw_target = get_swirl_num(n11_target, q11_target)

    blending_ratio = (Sw_target - Sw1)/(Sw2 - Sw1)

    return blending_ratio

def blend_from_moddel_cases(model_cases):

    for model_case in model_cases:

        #ファイルがあるかどうか確認
        if not os.path.exists(model_case.get_case_one_modelfile()):
            print(f"File not found: {model_case.get_case_one_modelfile()}")
            raise FileNotFoundError(f"The file not found: {model_case.get_case_one_modelfile()}")
        else:
            print(f"File found: {model_case.get_case_one_modelfile()}")

        if not os.path.exists(model_case.get_case_two_modelfile()):
            print(f"File not found: {model_case.get_case_two_modelfile()}")
            raise FileNotFoundError(f"The file not found: {model_case.get_case_two_modelfile()}")
        else:
            print(f"File found: {model_case.get_case_two_modelfile()}")
        

        feature_printer(json_feature_reader(model_case.get_case_one_modelfile()))
        feature_printer(json_feature_reader(model_case.get_case_two_modelfile()))

        json_feature_writer(
            model_case.get_output_modelfile(),
            json_feature_blender_1D(
                json_feature_reader(model_case.get_case_one_modelfile()),
                json_feature_reader(model_case.get_case_two_modelfile()),
                model_case.get_blending_ratio()
            )
        )

if __name__ == "__main__":


    test_case = Model_Case(
        case_one_modelfile="./FeatureBlender/src/20251104_GVO60_Traning Model.json",
        case_two_modelfile="./FeatureBlender/src/20251104_GVO80_Traning Model.json",
        output_modelfile="blended_model.json",
        blending_ratio=0.5
    )

    blend_from_moddel_cases([test_case])