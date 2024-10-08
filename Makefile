# Compilador
CXX = g++

# Flags de compilación
CXXFLAGS = -std=c++11 -Wall `pkg-config --cflags opencv4`

# Flags de enlazado
LDFLAGS = `pkg-config --libs opencv4`

# Nombre del ejecutable
TARGET = detector_rostros_serial

# Archivos fuente
SOURCES = main.cpp

# Regla por defecto
all: $(TARGET)

# Regla para compilar el ejecutable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Regla para limpiar archivos generados
clean:
	rm -f $(TARGET)

# Regla para ejecutar el programa
run: $(TARGET)
	./$(TARGET) imagen.jpeg

# Phony targets
.PHONY: all clean run
