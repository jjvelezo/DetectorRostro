#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>  // Incluir para OpenMP

void detectarRostrosSerial(const std::string& nombreArchivo) {
    // Cargar la imagen
    cv::Mat imagen = cv::imread(nombreArchivo);
    if (imagen.empty()) {
        std::cerr << "No se pudo cargar la imagen: " << nombreArchivo << std::endl;
        return;
    }

    // Cargar el clasificador de Haar Cascades para la detección de rostros
    cv::CascadeClassifier clasificadorRostros;
    if (!clasificadorRostros.load(cv::samples::findFile("haarcascade_frontalface_alt.xml"))) {
        std::cerr << "No se pudo cargar el clasificador Haar." << std::endl;
        return;
    }

    // Convertir la imagen a escala de grises
    cv::Mat imagenGris;
    cv::cvtColor(imagen, imagenGris, cv::COLOR_BGR2GRAY);

    // Detectar los rostros en la imagen
    std::vector<cv::Rect> rostros;
    clasificadorRostros.detectMultiScale(imagenGris, rostros);

    // Dibujar rectángulos verdes alrededor de los rostros detectados
    for (const auto& rostro : rostros) {
        cv::rectangle(imagen, rostro, cv::Scalar(0, 255, 0), 2);
    }

    // Guardar la imagen con los rostros detectados
    cv::imwrite("rostros_detectados_serial.jpg", imagen);

    // Guardar los rostros individualmente
    for (size_t i = 0; i < rostros.size(); ++i) {
        cv::Mat rostro = imagen(rostros[i]);
        std::string nombreRostro = "rostro_serial_" + std::to_string(i) + ".jpg";
        cv::imwrite(nombreRostro, rostro);
    }
}

void detectarRostrosParalelo(const std::string& nombreArchivo) {
    // Cargar la imagen
    cv::Mat imagen = cv::imread(nombreArchivo);
    if (imagen.empty()) {
        std::cerr << "No se pudo cargar la imagen: " << nombreArchivo << std::endl;
        return;
    }

    // Cargar el clasificador de Haar Cascades para la detección de rostros
    cv::CascadeClassifier clasificadorRostros;
    if (!clasificadorRostros.load(cv::samples::findFile("haarcascade_frontalface_alt.xml"))) {
        std::cerr << "No se pudo cargar el clasificador Haar." << std::endl;
        return;
    }

    // Convertir la imagen a escala de grises en paralelo
    cv::Mat imagenGris;
    #pragma omp parallelF
    {
        #pragma omp single
        cv::cvtColor(imagen, imagenGris, cv::COLOR_BGR2GRAY);
    }

    // Detectar los rostros en la imagen
    std::vector<cv::Rect> rostros;
    clasificadorRostros.detectMultiScale(imagenGris, rostros);

    // Dibujar rectángulos verdes alrededor de los rostros detectados
    for (const auto& rostro : rostros) {
        cv::rectangle(imagen, rostro, cv::Scalar(0, 255, 0), 2);
    }

    // Guardar la imagen con los rostros detectados
    cv::imwrite("rostros_detectados_parallel.jpg", imagen);

    // Guardar los rostros individualmente en paralelo
    #pragma omp parallel for
    for (size_t i = 0; i < rostros.size(); ++i) {
        cv::Mat rostro = imagen(rostros[i]);
        std::string nombreRostro = "rostro_parallel_" + std::to_string(i) + ".jpg";
        cv::imwrite(nombreRostro, rostro);
    }
}

int main() {
    std::string nombreArchivo;
    std::cout << "Ingrese el nombre de la imagen (JPEG): ";
    std::cin >> nombreArchivo;

    // Medir tiempo de ejecución en modo serial
    auto inicioSerial = std::chrono::high_resolution_clock::now();
    detectarRostrosSerial(nombreArchivo);
    auto finSerial = std::chrono::high_resolution_clock::now();
    double tiempoSerial = std::chrono::duration<double, std::milli>(finSerial - inicioSerial).count();

    // Medir tiempo de ejecución en modo paralelo
    auto inicioParalelo = std::chrono::high_resolution_clock::now();
    detectarRostrosParalelo(nombreArchivo);
    auto finParalelo = std::chrono::high_resolution_clock::now();
    double tiempoParalelo = std::chrono::duration<double, std::milli>(finParalelo - inicioParalelo).count();

    // Calcular la aceleración (speedup)
    double speedup = tiempoSerial / tiempoParalelo;

    // Mostrar los resultados de tiempo de ejecución
    std::cout << "---[RESULTADOS TIEMPO DE EJECUCIÓN]---\n";
    std::cout << "Tiempo de ejecución en modo serial: " << tiempoSerial << " ms\n";
    std::cout << "Tiempo de ejecución en modo paralelo: " << tiempoParalelo << " ms\n";
    std::cout << "Aceleración (speedup): " << speedup << "x\n";

    return 0;
}
