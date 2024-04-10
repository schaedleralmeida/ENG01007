# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio Grande do Sul
Departamento de Engenharia Civil
ENG01007 - Análise Estrutural por Computador
prof. Felipe Schaedler de Almeida

T2D_global: 
    Funções gerais do Método da Rigidez Direta (MRD)
    e específicas para análise de treliças
Data: Agosto de 2021
"""

import numpy as np
from math import *


class Material():
    '''
    Material elástico linear para análise estrutural por AECPy
    
    Atributes
    ----------
    E: float
        Módulo de Young
    G: float
        Módulo de cisalhamento
    cp: float
        Coeficiente de Poisson
    pe: float
        Peso específico
    cdt: float
        Coeficiente de dilatação térmica
    nome: str
        Nome do material
    '''

    def __init__(self,E,cp,pe=0,cdt=0,nome=""):
        '''
        Inicialização de uma instância da classe Material

        Parametres
        ----------
        E: float
            Módulo de Young
        cp: float
            Coeficiente de Poisson
        pe: float (optional)
            Peso específico (default 0)
        cdt: float (optional)
            Coeficiente de dilatação térmica (default 0)
        nome: str (optional)
            Nome do material (default "")

        Raises
        ------
        ValueError
            Se qualquer propriedade do material for negativa 
        '''

        if any([ prop<0 for prop in [E,cp,pe,cdt]]):
                raise ValueError('As propriedades do material não devem ser negativas')

        self.E = E      #módulo de Young (de elasticidade)
        self.cp = cp    #coeficiente de poisson
        self.G = E/(2*(1+cp)) #módulo de cisalhamento (de elasticidade transversal)
        self.pe = pe    #peso específico
        self.cdt = cdt  #Coefciente de dilatação térmica
        self.nome = nome #nome do material
        
class Secao():

    '''
    Seção transversal para um elemento de barra prismático
    usado na análise estrutural por AECPy

    Na denominação dos atributos, são considerados:
    eixo local 2 - eixo vertical
    eixo local 3 - eixo horizonal
    Os eixos 2 e 3 são eixos principais centrais de inércia da seção
    
    Atributes
    ----------
    mat: Material
        Material que forma o elemento estrutural
    A: float
        Área da seção transveral
    I2: float
        Momento de inércia em relação ao eixo local 2
    I3: float
        Momento de inércia em relação ao eixo local 3
    J: float
        Constante de torção pura (constante de St. Venant)
    AE: float
        Rigidez axial da seção
    EI2: float
        Rigidez da seção à flexão em torno do eixo local 2
    EI3: float
        Rigidez da seção à flexão em torno do eixo local 3 
    GJ: float
        Rigidez à torção pura da seção
    nome: str
        Nome do material
    peso_unitario
    r2
    r3
    '''

    def __init__(self,mat,A,I3=0,I2=0,J=0,nome=''):
        '''
        mat: Material
            Material que forma o elemento estrutural
        A: float
            Área da seção transveral
        I2: float (optional)
            Momento de inércia em relação ao eixo local 2. (deault é 0, aplicável à análise de treliças)
        I3: float (optional)
            Momento de inércia em relação ao eixo local 3 (default é 0, não aplicável para análise de pórticos espaciais)
        J: float (optional)
            Constante de torção pura (constante de St. Venant) (default é 0, não aplicável à análise de pórticos espaciais)
        nome: str (optional)
            Nome da seção transveral (default "")

        Raises
        ------
        ValueError
            Se qualquer propriedade geométrica for negativa
        '''
        
        if not isinstance(mat,Material):
            raise TypeError("mat deve ser do tipo Material")
        if any([ prop<0 for prop in [A,I2,I3,J]]):
                raise ValueError('As propriedades geométricas não devem ser negativas')

        self.mat = mat   #material 
        self.A = A      #Área da seção
        self.I2 = I2    #Momento de inérica da seção em relação ao eixo local 2
        self.I3 = I3    #Momento de inérica da seção em relação ao eixo local 3
        self.J = J      #Constante de torção pura (St. Venant)            
        self.nome = nome

        self.EA = mat.E*self.A     #Rigidez da seção ao alongamento/encurtamento axial
        self.EI2 = mat.E*self.I2   #Ridigez da seção à flexão em torno do eixo local 2 
        self.EI3 = mat.E*self.I3   #Ridigez da seção à flexão em torno do eixo local 3
        self.GJ = mat.G*self.J     #Rigidez da seção à torção pura (St. Venant)


    @property
    def peso_unitario(self):
        '''Peso por unidade de comprimento (da barra)'''
        return self.mat.pe * self.A
    
    @property
    def r2(self):
        '''Raio de giração em relação ao eixo 2'''
        return math.sqrt(self.I2/self.A)
    
    @property
    def r3(self):
        '''Raio de giração em relação ao eixo 3'''
        return math.sqrt(self.I3/self.A)



def T2D_L_phi(rI, rJ):
    """
    Calcula o comprimento (L) e o ângulo de inclinação (phi) em relação ao eixo 'x+'
    para um elemento em um problema plano (2D)
    
    Parameters
    ----------
    rI, rJ: numpy.ndarray
        Arrays com as coordenadas dos nós inicial (I) e final (J) do elemento
    """    

    #vetor posição relativa entre os nós da barra
    rIJ = rJ - rI
    #comprimento da barra
    L = np.linalg.norm(rIJ)
    #ângulo de inclinação da barra (em rad) em relação ao eixo x+
    phi = atan2(rIJ[1],rIJ[0])
    
    return L, phi

        

def T2D_K_global_phi(S,L,phi):
    
    """
    Calcula da matriz de rigidez global de uma barra de treliça plana (2D)
    em função do ângulo de inclinação da barra em relação ao eixo x (phi em radianos)
    
    Parameters
    ----------
    S: Secao
        Propriedades da seção transversal da barra
    L: float
        Comprimento da barra
    phi: float
        Ângulo entre o eixo x+ e o eixo da barra, em radianos, tomado positivo no sentido anti-horário
    """
    
    r = S.EA/L                #rigidez axial da seção
    sc = r*sin(phi)*cos(phi)  #sen * cos * EA/L
    c2 = r*cos(phi)**2        #cos * cos * EA/L 
    s2 = r*sin(phi)**2        #sen * sen * EA/L 

    K = np.zeros((4,4)) #matriz de rigidez 4x4 (iniciando com zeros)
    
    K[0,  : ] = [c2, sc, -c2, -sc] #1º linha
    K[1,  : ] = [sc, s2, -sc, -s2] #2º linha
    K[2,  : ] = - K[0,  : ]        #3º linha igual 1º linha vezes (-1)
    K[3,  : ] = - K[1,  : ]        #4º linha igual 2º linha vezes (-1)
    
    return K

            
