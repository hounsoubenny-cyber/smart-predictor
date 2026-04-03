#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:17:27 2026

@author: hounsousamuel
"""

class MyException(BaseException):
    def __init__(self, *args):
        super().__init__(*args)
    
def explain_exc(e:MyException):
    if not isinstance(e, MyException):
        print('[_EXPLAIN_EXC]Erreur : ', str(e))
        return
    
    else:
        txt = str(getattr(e, "txt", ""))
        cop = ""
        if txt:
            print("Une erreur est survenue : \n")
            cop = txt.lower()
            if cop.startswith("clé absente"):
                print("Une clé du dictionaire de configuration est absente, il s'agit de la clé entre parenthèse.")
                print('Texte de l\'erreur : \n', txt)
                return
            
            elif cop.startswith("config vide"):
                print('Le dictionnaire de configuaration est vide.\nTexte de l\'erreur : \n', txt)
                return
            
            elif cop.startswith("valeur vide"):
                print("Une clé du dictionnaire est à une valeur vide alors qu'elle ne devrait pas, la clé entre parenthèse !")
                print('Texte de l\'erreur : \n', txt)
            
            elif cop.startswith("target name"):
                print('Veuillez renseignez la target_name dans les configurations !')
                print('Texte de l\'erreur : \n', txt)
            