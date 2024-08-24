import dearpygui.dearpygui as dpg

some_dict = {
    'some_string': None
}

def print_some_dict():
    print(some_dict)
    
def add(_, app_data, user_data):
    some_dict[user_data] = app_data

dpg.create_context()

with dpg.window(tag="Primary Window", label="Example Window"):
    dpg.add_text("Hello world")
    dpg.add_button(label="Save", callback=print_some_dict)
    dpg.add_input_text(label="string", callback=add, user_data='some_string')
    dpg.add_slider_float(label="float")
    dpg.add_combo(label="Combobox", items=["foo", "bar"], callback=add, user_data="some_config_item")

dpg.create_viewport(title='Custom Title', width=400, height=600, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()