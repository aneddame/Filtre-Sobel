#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Chemin de l'image
    std::string imagePath = "/home/odroid/Desktop/flcss/img/right/3.jpeg";

    // Charger l'image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Vérifier si l'image a été chargée correctement
    if (image.empty()) {
        std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
        return -1;
    }

    // Convertir l'image en niveaux de gris
    cv::Mat imageGrayscale;
    cv::cvtColor(image, imageGrayscale, cv::COLOR_BGR2GRAY);

    // Créer des matrices pour les gradients
    cv::Mat grad_x = cv::Mat::zeros(imageGrayscale.size(), CV_16S);
    cv::Mat grad_y = cv::Mat::zeros(imageGrayscale.size(), CV_16S);

    // Noyau Sobel horizontal (Gx)
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Noyau Sobel vertical (Gy)
    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Appliquer les filtres Sobel avec une boucle for
    for (int i = 1; i < imageGrayscale.rows - 1; i++) {
        for (int j = 1; j < imageGrayscale.cols - 1; j++) {
            int sum_x = 0, sum_y = 0;

            // Convolution pour le gradient horizontal (Gx)
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel = imageGrayscale.at<uchar>(i + k, j + l);
                    sum_x += sobel_x[k + 1][l + 1] * pixel;
                    sum_y += sobel_y[k + 1][l + 1] * pixel;
                }
            }

            // Stocker les résultats dans les matrices de gradients
            grad_x.at<short>(i, j) = sum_x;
            grad_y.at<short>(i, j) = sum_y;
        }
    }

    // Convertir les résultats en type CV_8U pour l'affichage
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Combiner les résultats des gradients horizontaux et verticaux
    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Afficher l'image originale, les gradients horizontaux et verticaux
    cv::imshow("Image originale", imageGrayscale);
    cv::imshow("Gradient horizontal (Gx)", abs_grad_x);
    cv::imshow("Gradient vertical (Gy)", abs_grad_y);
    cv::imshow("Gradient combiné", grad);

    // Attendre que l'utilisateur appuie sur une touche
    cv::waitKey(0);

    return 0;
}
