# model free prediciton system.
# Strawberry_1 is always the goal.

# Order the mushrooms:
straw_1, straw_2, straw_3 = heuristic_order_mushrooms(start_environment_state) 

straw_1_past = False
straw_2_past = False
straw_3_past = False

for index, item in enumerate(trajectory):
	# 1. Read current state of system.
	straw_1_state_pos, straw_1_state_ori = get_state(straw_1)
	straw_2_state_pos, straw_2_state_ori = get_state(straw_2)
	straw_3_state_pos, straw_3_state_ori = get_state(straw_3)
	# 2. predict new state given next 10 actions
	current_traj = trajectory[index:(index + 10)]
	straw_1_predicted_state, straw_2_predicted_state, straw_3_predicted_state = DNN(straw_1_state_pos, straw_1_state_ori,
																					straw_2_state_pos, straw_2_state_ori,
																					straw_3_state_pos, straw_3_state_ori,
																					current_traj)
	# 3. Calculate Errors:
	if straw_3_past = False:
		st_x_error = straw_3_state_pos.x - straw_3_predicted_state.x	
		st_y_error = straw_3_state_pos.y - straw_3_predicted_state.y
		st_z_error = straw_3_state_pos.z - straw_3_predicted_state.z
	elif straw_3_past = False:
		st_x_error = straw_2_state_pos.x - straw_2_predicted_state.x	
		st_y_error = straw_2_state_pos.y - straw_2_predicted_state.y
		st_z_error = straw_2_state_pos.z - straw_2_predicted_state.z
	else:
		st_x_error = straw_1_state_pos.x - straw_1_predicted_state.x	
		st_y_error = straw_1_state_pos.y - straw_1_predicted_state.y
		st_z_error = straw_1_state_pos.z - straw_1_predicted_state.z

	# 4. Ensure current strawberry is at goal state:



