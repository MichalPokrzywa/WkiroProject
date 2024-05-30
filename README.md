# Rozpoznawanie skóry na zdjęciach przy pomocy własnej implementacji naiwnego klasyfikatora Bayesa

## Autorzy
- Maciej Gromek
- Michał Pokrzywa
- Wojciech Omelańczuk

## Temat
Rozpoznawanie skóry na zdjęciach przy pomocy własnej implementacji naiwnego klasyfikatora Bayesa.

## Zakres projektu
1. Własna implementacja klasyfikatora naiwnego Bayesa.
2. Analiza zależności pomiędzy błędem detekcji skóry (metryka zbalansowana, np. F-miarą) a wielkością zbioru uczącego oraz liczbą przedziałów histogramu (palety kanałów RGB).
3. Testowanie klasyfikatora na pomniejszonych zdjęciach.

## Implementacja klasyfikatora
Projekt został zaimplementowany w języku Python. Główna klasa `NaiveBayesClassifier` jest przeznaczona do klasyfikacji pikseli na skórne i nieskórne. Klasa obejmuje następujące etapy:

### 1. Inicjalizacja
Konstruktor klasy `__init__` przyjmuje cztery argumenty:
- `images`: lista obrazów wejściowych.
- `images_skin`: lista masek skóry odpowiadających obrazom wejściowym.
- `images_test`: lista obrazów testowych.
- `images_test_mask`: lista masek testowych dla obrazów testowych.

Podczas inicjalizacji obrazy są konwertowane do przestrzeni kolorów YCrCb.

### 2. Separacja pikseli skóry
Metoda `separate_skin_pixels` oddziela piksele skóry od nieskórnych na podstawie dostarczonych masek skóry, tworząc listy `skin_pixels` i `non_skin_pixels`.

### 3. Estymacja funkcji gęstości prawdopodobieństwa
Metoda `estimate_pdf` szacuje funkcję gęstości prawdopodobieństwa (PDF) dla pikseli używając histogramu wielowymiarowego.

### 4. Znajdowanie indeksu kosza
Metoda `find_bin_index` służy do znajdowania indeksu kosza w histogramie dla danego piksela na podstawie wartości granicznych histogramu.

### 5. Obliczanie prawdopodobieństwa warunkowego klasy
Metoda `class_conditional_prob` oblicza prawdopodobieństwo warunkowe dla piksela używając wartości histogramu.

### 6. Obliczanie prawdopodobieństwa a posteriori
Metoda `posterior_prob` oblicza prawdopodobieństwo a posteriori, że dany piksel należy do klasy skóry.

### 7. Klasyfikacja obrazu
Metoda `classify_image` klasyfikuje piksele obrazu na skórne lub nieskórne na podstawie obliczonego prawdopodobieństwa a posteriori.

### 8. Tworzenie mapy skóry
Metoda `create_skin_map` wykonuje cały proces uczenia i testowania klasyfikatora, zwracając wartości metryk jakości klasyfikacji (precyzja, recall, F1 score) oraz wygenerowaną maskę skóry.

## Wyniki i wnioski
- **Błąd detekcji maleje wraz ze wzrostem liczby obrazów.**
- **Większe koszyki histogramów (więcej bins) prowadzą do mniejszego błędu detekcji.**
- **Zmniejszenie zdjęć prowadzi do wyższego błędu detekcji.**

## Wykresy
- **Wykres 1:** Błąd detekcji w zależności od liczby obrazów.
- **Wykres 2:** Błąd detekcji w zależności od rozmiaru koszyka histogramu.
Obserwacje z wykresów wykazują, że większa liczba obrazów i większe koszyki histogramów poprawiają dokładność detekcji skóry, natomiast zmniejszenie rozmiaru zdjęć wpływa negatywnie na precyzję klasyfikacji.
![Diagram](https://github.com/MichalPokrzywa/WkiroProject/blob/main/wykres.png)



## Link do repozytorium
[GitHub - WkiroProject](https://github.com/MichalPokrzywa/WkiroProject)

## Bibliografia
- J. N. J. K. Michal Kawulok, „Skin Detection and Segmentation in Color Images,” Springer Science+Business Media Dordrecht, pp. 329-366, 2014.
