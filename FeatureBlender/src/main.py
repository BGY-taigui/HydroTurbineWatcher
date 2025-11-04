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

def json_feature_blender_2D(feature_data_1: dict, feature_data_2: dict, blending_ratio: float):

    blended_feature_data = feature_data_1

    if len(feature_data_1["Feature Manipulation"]) !=1 and len(feature_data_2["Feature Manipulation"]) != 1:
        raise ValueError("Feature Manipulation length is not 1 in one of the feature data.")

    blended_feature_data["Feature Manipulation"][0]["mean values"] = (np.array(feature_data_1["Feature Manipulation"][0]["mean values"])* blending_ratio + np.array(feature_data_2["Feature Manipulation"][0]["mean values"])* (1 - blending_ratio)).tolist()

    blended_feature_data["Feature Manipulation"][0]["standard deviation values"] = (np.array(feature_data_1["Feature Manipulation"][0]["standard deviation values"])* blending_ratio + np.array(feature_data_2["Feature Manipulation"][0]["standard deviation values"])* (1 - blending_ratio)).tolist()

    blended_feature_data["Anomaly Detection"]["mean values"] = (np.array(feature_data_1["Anomaly Detection"]["mean values"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["mean values"]) * (1 - blending_ratio)).tolist()


    # Blending eigen vectors
    eigenvectors_1 = feature_data_1["Anomaly Detection"]["eigenvectors"]
    eigenvectors_2 = feature_data_2["Anomaly Detection"]["eigenvectors"]
    blended_eigenvectors = []
    for ev1, ev2 in zip(eigenvectors_1, eigenvectors_2):
        blended_eigenvector = (np.array(ev1) * blending_ratio + np.array(ev2) * (1 - blending_ratio)).tolist()
        blended_eigenvectors.append(blended_eigenvector)

    blended_feature_data["Anomaly Detection"]["eigenvectors"] = blended_eigenvectors

    blended_feature_data["Anomaly Detection"]["variance"] = (np.array(feature_data_1["Anomaly Detection"]["variance"]) * blending_ratio + np.array(feature_data_2["Anomaly Detection"]["variance"]) * (1 - blending_ratio)).tolist()

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

if __name__ == "__main__":

    case_one_modelfile = "./FeatureBlender/TestCase/20241008_111458_1650_Traning Model.json"
    case_two_modelfile = "./FeatureBlender/TestCase/20241115_111458_1650_Traning Model.json"


    blending_ratio = 0.5


    feature_printer(json_feature_reader(case_one_modelfile))
    feature_printer(json_feature_reader(case_two_modelfile))

    json_feature_writer(
        "./test.json",
        json_feature_blender_2D(
            json_feature_reader(case_one_modelfile),
            json_feature_reader(case_two_modelfile),
            blending_ratio
        )
    )
