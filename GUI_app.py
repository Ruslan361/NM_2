#!/bin/sh
#!/bin/bash
#!/bin/env python
#!/usr/bin/env rdmd
# -*- coding: utf-8 -*-
# ./.venv/bin/python
#GUI application

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import sys
import traceback
from resolve import MainTask, TestTask, method_balance
import math

# In[]:

df_tmp = pd.DataFrame({
    '     1     ': [],
    '     2     ': [],
    '       3       ': [],
    '       4       ': [],
})

fig, graf = plt.subplots(figsize=(5, 5))
# graf = plt.plot([1, 2, 3, 4], [1, 2, 3, 4])

list_q = ['Тестовая', 'Основная']

frame_output_data = [
    [sg.Output(size=(100, 17), key="-DATA-")],
    [sg.Button("Clear", size=(100, 1))],
]

# In[]:
column1 = [
    [sg.DropDown(list_q, default_value=list_q[0], size=(25, 1), key='-SELECTOR-')],
    [sg.Text("Количество участков разбиений", size=(50, 1))],
    [sg.InputText(default_text="2", size=(27, 1), key='-N-')],
    [sg.Submit(size=(24, 1))],
    [sg.Exit(size=(24, 1))]
]

column_table = [
    [sg.Table(values=df_tmp.values.tolist(), headings=df_tmp.columns.tolist(),
            alternating_row_color='darkblue', key='-TABLE-', vertical_scroll_only = False,
            row_height = 25, size=(500, 19), justification='left')],
]

column_graf = [
    [sg.Canvas(key="-CANVAS-")],
]

# In[]:
layout = [
    [sg.Column(column1), sg.VerticalSeparator(), sg.Frame("Данные", frame_output_data, element_justification='right')],
    [sg.HorizontalSeparator()],
    [sg.Column(column_table, key='-SIZETABLE-'), sg.VerticalSeparator(), sg.Column(column_graf, size=(500, 500), justification='right', key='-SIZEGRAF-')],
]

# In[]:
window = sg.Window('LAB2', layout, finalize=True, resizable=True, grab_anywhere=True)

last_w_size = window.size

canvas_elem = window["-CANVAS-"]
canvas = FigureCanvasTkAgg(fig, master=canvas_elem.Widget)
canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

def update_title(table, headings):
    for cid, text in zip(df_tmp.columns.tolist(), headings):
        table.heading(cid, text=text)

def on_resize(event):
    if (last_w_size != window.size):
        width, height = window.size
        window.Element('-SIZETABLE-').set_size((window.size[0]/2-50, window.size[1]))
        window.Element('-TABLE-').set_size((500, window.size[1]))
        window.Element('-SIZEGRAF-').set_size((window.size[0]/2, window.size[1]/1.4))
        window.Element('-CANVAS-').set_size((window.size[0]/2, window.size[1]/1.4))
        canvas_elem.Widget.pack(side="top", fill="both", expand=True)
        window.FindElement('-DATA-').Update('')
        
window.TKroot.bind('<Configure>', on_resize)

while True:                             # The Event Loop
    event, values = window.read()
    window.FindElement('-DATA-').Update('')
    # print(event, values) #debug
    # try:
    if event in (None, 'Exit', 'Cancel'):
        break
    if event == 'Submit':
        selector = values['-SELECTOR-']
        sel = list_q.index(selector)+1
        if sel == 1:
            task = TestTask()
        else:
            task = MainTask()

        n = int(window.Element("-N-").Get())
        
        x, v = method_balance(n, task)

        df = pd.DataFrame({
            'x': x,
            'v': v,
        })

        eps = 0
        id = 0

        if sel == 1:
            list_u = [task.u(xi) for xi in x]
            df.insert(loc = len(df.columns), column = 'u', value = list_u)
            df.insert(loc = len(df.columns), column = '|u - v|', value = [math.fabs(ui - vi) for ui, vi in zip(list_u, v)])

            eps = max([math.fabs(ui - vi) for ui, vi in zip(list_u, v)])
            id = df[df["|u - v|"] == eps].index[0]

            graf = plt.plot(df['x'], df['v'], 'b-')
            graf = plt.plot(df['x'], df['u'], 'r-')
        else:
            x2, v2 = method_balance(2*n, task)
            v2_2 = [v2[i] for i in range(0, len(v2), 2)]
            df.insert(loc = len(df.columns), column = 'v2', value = v2_2)
            df.insert(loc = len(df.columns), column = '|v2 - v|', value = [math.fabs(v2i - vi) for v2i, vi in zip(v2_2, v)])

            eps = max([math.fabs(v2i - vi) for v2i, vi in zip(v2_2, v)])
            id = df[df["|v2 - v|"] == eps].index[0]

            graf = plt.plot(x, v, 'b-')
            graf = plt.plot(x, v2_2, 'r-')

        table = window.Element("-TABLE-").Widget

        update_title(table, df.columns.tolist())
        
        window.Element("-TABLE-").Update(values = df.values.tolist())

        print("Вариант 1, Точка разрыва = 0.4, Диапозон = [0, 1]," + "\nколичество узлов = " + str(n+1) +  ",\nШаг сетки  = " + str(1/n) + "\n"
              + "Максимальная погрешность = " + str(eps) + ", на шаге " + str(id))

        canvas.draw()
    if event == "Clear":
        fig.clear()
        canvas.draw()
    # except Exception as e:
    #         exc_type, exc_value, exc_traceback = sys.exc_info()

# traceback.print_exception(exc_type, exc_value, exc_traceback)
window.close()