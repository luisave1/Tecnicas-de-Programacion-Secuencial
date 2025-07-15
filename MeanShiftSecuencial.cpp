//=======================MEAN SHIFT CON MPI====================
//==============ALGORITMOS PARALELOS DISTRIBUIDOS===========
//============L. Fernando Cc.==============================
#include <chrono>               // Para medir el tiempo de ejecución
#include <iostream>             // Para imprimir en consola
#include <opencv2/opencv.hpp>   // Librería OpenCV para procesamiento de imágenes
#include <cmath>                // Para funciones matemáticas como sqrt

using namespace cv;
using namespace std;

// ==== Parámetros del algoritmo Mean Shift ====
const float hs = 8.0f;          // Radio espacial (tamaño del vecindario)
const float hr = 16.0f;         // Radio de color (distancia máxima en el espacio Lab)
const int maxIter = 5;          // Número máximo de iteraciones por píxel
const float tol_color = 0.3f;   // Tolerancia mínima de cambio en color
const float tol_spatial = 0.3f; // Tolerancia mínima de cambio en posición

// ==== Estructura para representar un punto en 5 dimensiones (espacio + color) ====
struct Point5D {
    float x, y, l, a, b;

    Point5D() : x(0), y(0), l(0), a(0), b(0) {}
    Point5D(float x_, float y_, float l_, float a_, float b_)
        : x(x_), y(y_), l(l_), a(a_), b(b_) {}

    // Distancia en el espacio de color (Lab)
    float colorDist(const Point5D& p) const {
        return sqrt((l - p.l) * (l - p.l) + (a - p.a) * (a - p.a) + (b - p.b) * (b - p.b));
    }

    // Distancia en el espacio espacial (x, y)
    float spatialDist(const Point5D& p) const {
        return sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y));
    }

    // Suma de dos puntos 5D
    Point5D operator+(const Point5D& p) const {
        return Point5D(x + p.x, y + p.y, l + p.l, a + p.a, b + p.b);
    }

    // División de un punto 5D por un escalar (promedio)
    Point5D operator/(float val) const {
        return Point5D(x / val, y / val, l / val, a / val, b / val);
    }
};

// Convierte un píxel (i, j) de la imagen en un punto 5D en Lab
Point5D getPoint5D(int i, int j, const Mat& labImg) {
    Vec3b color = labImg.at<Vec3b>(i, j);
    return Point5D((float)j, (float)i,
        color[0] * 100.0f / 255.0f,   // Escalado de L
        (float)color[1] - 128.0f,     // Desplazamiento de a
        (float)color[2] - 128.0f);    // Desplazamiento de b
}

// === Algoritmo Mean Shift aplicado sobre la imagen en Lab ===
void applyMeanShift(Mat& labImg) {
    int rows = labImg.rows, cols = labImg.cols;

    // Recorre cada píxel de la imagen
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            Point5D current = getPoint5D(y, x, labImg);  // Punto actual
            Point5D prev;

            int iter = 0;
            // Itera hasta que se cumpla el criterio de convergencia o el máximo de iteraciones
            do {
                prev = current;
                Point5D sum(0, 0, 0, 0, 0);  // Suma acumulada de puntos vecinos
                int count = 0;

                // Recorre vecindario definido por hs
                for (int j = -hs; j <= hs; ++j) {
                    for (int i = -hs; i <= hs; ++i) {
                        int nx = x + i;
                        int ny = y + j;

                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            Point5D neighbor = getPoint5D(ny, nx, labImg);
                            if (current.spatialDist(neighbor) <= hs &&
                                current.colorDist(neighbor) <= hr) {
                                sum = sum + neighbor;
                                count++;
                            }
                        }
                    }
                }

                if (count > 0) {
                    current = sum / count;  // Nuevo centro
                }

                iter++;
            } while (current.colorDist(prev) > tol_color &&
                     current.spatialDist(prev) > tol_spatial &&
                     iter < maxIter);

            // Convertir de nuevo a formato Lab de OpenCV
            int l = static_cast<int>(current.l * 255.0f / 100.0f);
            int a = static_cast<int>(current.a + 128.0f);
            int b = static_cast<int>(current.b + 128.0f);
            labImg.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(l),
                                           saturate_cast<uchar>(a),
                                           saturate_cast<uchar>(b));
        }
    }
}

// === Función principal ===
int main() {
    // Cargar imagen original
    Mat Img = imread("C:/Users/LUIS FERNANDO/Pictures/arte/THL.jpg");
    if (Img.empty()) {
        cerr << "Error: No se pudo abrir o encontrar la imagen 'THL.jpg'" << endl;
        return -1;
    }

    // Redimensionar imagen para pruebas
    resize(Img, Img, Size(256, 256), 0, 0, INTER_LINEAR);

    // Mostrar imagen original
    namedWindow("The Original Picture");
    imshow("The Original Picture", Img);

    // Medir tiempo de ejecución
    auto start = chrono::high_resolution_clock::now();

    // Convertir de BGR a Lab
    cvtColor(Img, Img, COLOR_BGR2Lab);
    applyMeanShift(Img);  // Aplicar el algoritmo

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "Tiempo de ejecución: " << duration.count() << " ms" << endl;

    // Convertir de nuevo a BGR para mostrar
    cvtColor(Img, Img, COLOR_Lab2BGR);

    // Mostrar imagen resultante
    namedWindow("MS Picture");
    imshow("MS Picture", Img);

    waitKey(0);
    return 0;
}
