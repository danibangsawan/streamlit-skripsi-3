# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Fungsi untuk memuat dan menampilkan data
def load_data():
    data = pd.read_csv('corrected_data_kuesioner.csv', sep=';')
    return data

# Fungsi untuk visualisasi confusion matrix
def plot_confusion_matrix(y_test, prediksi_svm):
    cm_svm = confusion_matrix(y_test, prediksi_svm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Prediksi -1', 'Prediksi 1'], yticklabels=['Asli -1', 'Asli 1'])
    plt.xlabel('Prediksi')
    plt.ylabel('Asli')
    plt.title('Confusion Matrix')
    st.pyplot()

# Fungsi untuk menampilkan metrik evaluasi
def display_evaluation_metrics(y_test, prediksi_svm):
    akurasi_svm = accuracy_score(y_test, prediksi_svm)
    precision_svm = precision_score(y_test, prediksi_svm)
    recall_svm = recall_score(y_test, prediksi_svm)
    f1_svm = f1_score(y_test, prediksi_svm)
    
    st.write(f"Akurasi: {akurasi_svm:.2f}")
    st.write(f"Precision: {precision_svm:.2f}")
    st.write(f"Recall: {recall_svm:.2f}")
    st.write(f"F1-Score: {f1_svm:.2f}")
    
    st.write("### Classification Report:")
    st.text(classification_report(y_test, prediksi_svm, target_names=['Kelas -1', 'Kelas 1']))

# Fungsi untuk mempersiapkan model dan melakukan training
def train_svm_model():
    data_latih = pd.read_csv('data_latih.csv')
    data_uji = pd.read_csv('data_uji.csv')
    
    X_train = data_latih.drop(columns='Label')
    y_train = data_latih['Label']
    X_test = data_uji.drop(columns='Label')
    y_test = data_uji['Label']
    
    model_svm = SVC(kernel='linear')
    model_svm.fit(X_train, y_train)
    
    prediksi_svm = model_svm.predict(X_test)
    
    return y_test, prediksi_svm

# Fungsi untuk visualisasi tingkat kepuasan
def plot_kepuasan(df):
    # Kategori Kepuasan
    df['Rata_rata'] = df[[f'KP{i}' for i in range(1, 11)]].astype(int).mean(axis=1)
    df['Kepuasan'] = df['Rata_rata'].apply(lambda x: 'Puas' if x >= 3.5 else 'Tidak Puas')
    
    # Jumlah kepuasan untuk setiap aplikasi
    kepuasan_counts = df.groupby(['Aplikasi', 'Kepuasan']).size().unstack(fill_value=0)
    
    st.write("### Jumlah Kepuasan untuk Setiap Aplikasi")
    st.write(kepuasan_counts)
    
    # Persentase Kepuasan untuk Setiap Aplikasi
    kepuasan_percent = kepuasan_counts.div(kepuasan_counts.sum(axis=1), axis=0) * 100
    st.write("### Persentase Kepuasan untuk Setiap Aplikasi (%)")
    st.write(kepuasan_percent.round(2))
    
    # Visualisasi Diagram Batang Kepuasan
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    index = range(len(kepuasan_percent))
    
    bar1 = ax.bar(index, kepuasan_percent["Puas"], bar_width, label="Puas")
    bar2 = ax.bar([i + bar_width for i in index], kepuasan_percent["Tidak Puas"], bar_width, label="Tidak Puas")
    
    for i, rect in enumerate(bar1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, f'{kepuasan_percent["Puas"][i]:.2f}%', ha='center', va='bottom', fontsize=10)
    
    for i, rect in enumerate(bar2):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, f'{kepuasan_percent["Tidak Puas"][i]:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Aplikasi')
    ax.set_ylabel('Persentase (%)')
    ax.set_title('Tingkat Kepuasan Pengguna')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(kepuasan_percent.index)
    plt.ylim(0, 100)
    ax.legend()
    st.pyplot()

# Fungsi untuk menampilkan ukuran data latih dan uji
def plot_data_size():
    data_latih = pd.read_csv('data_latih.csv')
    data_uji = pd.read_csv('data_uji.csv')

    train_size = data_latih.shape[0]
    test_size = data_uji.shape[0]
    total_size = train_size + test_size

    train_percent = (train_size / total_size) * 100
    test_percent = (test_size / total_size) * 100

    st.write(f"Jumlah Data Latih: {train_size} ({train_percent:.2f}%)")
    st.write(f"Jumlah Data Uji: {test_size} ({test_percent:.2f}%)")

    # Visualisasi diagram perbandingan ukuran data latih dan uji
    sizes = [train_size, test_size]
    labels = ['Data Latih', 'Data Uji']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, sizes, color=['blue', 'green'])
    ax.set_title('Perbandingan Ukuran Data Latih dan Data Uji')
    ax.set_ylabel('Jumlah Sampel')

    for bar, size, percent in zip(bars, sizes, [train_percent, test_percent]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{size} ({percent:.2f}%)', ha='center', va='bottom', fontsize=10)

    st.pyplot()

# Sidebar
st.sidebar.title("Menu Aplikasi")
sidebar_option = st.sidebar.radio(
    "Pilih Halaman:",
    ('Data', 'Model dan Evaluasi', 'Visualisasi Kepuasan')
)

# Halaman Data
if sidebar_option == 'Data':
    st.title("Data Kuesioner")
    data = load_data()
    st.write("### Data Kuesioner (5 Baris Teratas):")
    st.dataframe(data.head())
    
    # Tampilkan ukuran data latih dan uji
    plot_data_size()

# Halaman Model dan Evaluasi
elif sidebar_option == 'Model dan Evaluasi':
    st.title("Model SVM dan Hasil Evaluasi")
    st.write("Melatih model SVM menggunakan data latih...")
    
    y_test, prediksi_svm = train_svm_model()
    
    st.write("### Confusion Matrix")
    plot_confusion_matrix(y_test, prediksi_svm)
    
    st.write("### Metrik Evaluasi")
    display_evaluation_metrics(y_test, prediksi_svm)

# Halaman Visualisasi Kepuasan
elif sidebar_option == 'Visualisasi Kepuasan':
    st.title("Visualisasi Tingkat Kepuasan Pengguna")
    data = load_data()
    plot_kepuasan(data)
