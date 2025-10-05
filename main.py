import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# --- Load your models (adjust paths as needed) ---
classifier_model = load_model('./models/MRI-histopathological-classifier.keras')
histopathological_model = load_model('./models/resnet50_cancer_model-finetuned-version-1.keras')
MRI_model = load_model('./models/resnet50_cancer_model-MRI-finetuned-version-1.keras')

histo_class_names =['Acute Lymphoblastic Leukemia_early', 'Acute Lymphoblastic Leukemia_normal', 'Acute Lymphoblastic Leukemia_pre', 'Acute Lymphoblastic Leukemia_pro', 'breast_malignant', 'breast_normal', 'lung_colon_Adenocarcinoma', 'lung_colon_normal', 'lung_squamous cell carcinoma']
MRI_class_names = ['brain_glioma_tumor', 'brain_meningioma_tumor', 'brain_normal', 'brain_pituitary_tumor', 'kidney_cyst', 'kidney_normal', 'kidney_stone', 'kidney_tumor', 'pancreatic_normal', 'pancreatic_tumor']

# --- Grad-CAM function for Streamlit ---
def generate_gradcam(model, img_array, img_path, class_index, class_names, confidence, layer_name='conv5_block3_out'):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    # Invert heatmap
    heatmap_resized = 1 - heatmap_resized
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)


    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)

    # Combine original and Grad-CAM images side by side
    combined_img = np.hstack((original_rgb, superimposed_img))

    st.image(combined_img, caption=f"Original | Grad-CAM: {class_names[class_index]} (Confidence: {confidence:.2f})", use_container_width=True)

# --- Streamlit UI ---
st.title("Cancer Classifier with CNN Resnet50 and Grad-CAM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to a temporary path
    img_path = f"./temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    # --- Classification routing ---
    pred = classifier_model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]

    if pred_class == 1:  # histopathological
        st.write("Classified as histopathological.")
        result = histopathological_model.predict(img_array)
        result_class = np.argmax(result, axis=1)[0]
        confidence = result[0][result_class]
        generate_gradcam(histopathological_model, img_array, img_path, result_class, histo_class_names, confidence)
    else:  # MRI
        st.write("Classified as MRI.")
        result = MRI_model.predict(img_array)
        result_class = np.argmax(result, axis=1)[0]
        confidence = result[0][result_class]
        generate_gradcam(MRI_model, img_array, img_path, result_class, MRI_class_names, confidence)
