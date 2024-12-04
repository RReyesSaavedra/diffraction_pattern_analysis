class Muestra:
    def __init__(self, id_muestra, molaridad, intensidad):
        self.id_muestra = id_muestra
        self.molaridad = molaridad
        self.intensidad = intensidad

    def __repr__(self):
        return f"Muestra(id_muestra={self.id_muestra}, molaridad={self.molaridad}, intensidad={self.intensidad})"

# Función que genera un arreglo (lista) de objetos
def crear_muestras():
    muestras = [
        Muestra(1, 0.0, 100),
        Muestra(2, 0.1875, 200),
        Muestra(3, 0.375, 150),
        Muestra(4, 0.75, 250),
        Muestra(5, 1.5, 300),
        Muestra(6, 3.0, 400),
    ]
    return muestras

# Uso de la función
muestras = crear_muestras()
for muestra in muestras:
    print(muestra)
