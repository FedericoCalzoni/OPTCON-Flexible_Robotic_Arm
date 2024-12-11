# OPTCON-Flexible_Robotic_Arm

## Todos

- [ ] riordinare il codice:
    - [ ] spostare clcolo equilibri su una funzione/file dedicato
    - [ ] usare parameters.py in maniera più "corretta"
    - [ ] commentare (in inglese) chatgpt prompt: make docsrings for these methods in google style.
    - [ ] plot delle reference è stato messo un po' a caso dentro reference_trajectory
- [ ] implementare armijo nella root finding routine
- [ ] implementare plot delle reference ottenute dai due equilibri
    - [ ] parametri messi in parameters.py
- [ ] implementare una funzione di costo
    - [ ] parametri messi in parameters.py
- [ ] implementare una funzione che calcoal il gradiente della funzione di costo (use sympy)
    - [ ] nell aggiornamento della funzione di costo non utilizzare lo shooting method ma il closed loop update.
- [ ] usare soluzione (affine) LQR per calcolare la direzione di discesa.
FEDE SEI BELLO