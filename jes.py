from jes_sim import Sim
from jes_ui import UI
from utils import read_config

sim_config_file: str = 'config/sim_config.json'
ui_config_file: str = 'config/ui_config.json'

sim_config: dict = read_config(filename=sim_config_file)
ui_config: dict = read_config(filename=ui_config_file)

c_input = 500

sim: Sim = Sim(creature_count=int(c_input), config=sim_config)

# Cosmetic UI variables

ui: UI = UI(config=ui_config)

sim.ui = ui
ui.sim = sim
ui.add_buttons_and_sliders()
    
sim.initialize_universe()

while ui.running:
    sim.check_alap()
    ui.detect_mouse_motion()
    ui.detect_events()
    ui.detect_sliders()
    ui.do_movies()
    ui.draw_menu()
    ui.show()