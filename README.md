# pyKdV
Solves Korteweg–De Vries (KdV) equation (commonly used to model water waves) in a 1D periodic domain:

<img width="271" alt="Screen Shot 2022-04-28 at 3 45 28 PM" src="https://user-images.githubusercontent.com/33580851/165842620-acf11f8f-6bfe-43d0-bcdd-6939277ad10e.png">

where the subscripts *t* and *x* refer to temporal and spatial derivatives of the field *u*.

# Installation details

Use the command below to install the prerequisites:

```{sh}
pip install -r requirements.txt
```

Below is an illustration of an initialised cosine wave evolving into a train of solitons with time:

https://user-images.githubusercontent.com/33580851/165839882-b575faad-1613-45ec-b86f-d016bdcdfa2b.mp4

# Spatial convergence of the solver

<img width="918" alt="Screen Shot 2022-04-28 at 3 47 00 PM" src="https://user-images.githubusercontent.com/33580851/165842824-867092db-0812-4413-ae3f-0fca5f70b3ce.png">

# Temporal convergence of the solver

<img width="956" alt="Screen Shot 2022-04-28 at 3 47 47 PM" src="https://user-images.githubusercontent.com/33580851/165842945-fbe93df7-e059-4f7a-bcca-2b6971f2b497.png">
