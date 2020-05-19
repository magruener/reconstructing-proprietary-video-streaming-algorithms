# MAKE WILL SAVE THE WORLD :)
# credits: Licciardello Melissa, for any question do not hesitate to contact me
# email: melissa.licciardello@inf.ethz.ch
# Affiliation: ETH Zurich
# Favourite food: Pizza Regina Margherita, but anything that is vegetariand and doesn't taste like cultural appropriation is fine 


.PHONY: all clean
SHELL := /bin/bash

testcases_path = testcases
results_path = results
server_path = src/video_server

traces_path = traces_ATC
traces = $(sort $(notdir $(wildcard $(traces_path)/*)))

abrs = RB BB BOLA robustMPC robustMPC_three RL robustmpc_tree arte_tree fandom_tree twitch_tree tubitv_tree rl_tree robustmpc_complex_tree rl_complex_tree zdf_tree xvideos_tree srf_tree robustmpc_raw_tree robustmpc_automated_tree

abrs_figures_validation = robustmpc_tree robustmpc_raw_tree robustmpc_complex_tree robustmpc_automated_tree robustMPC_three # rl_tree  rl_complex_tree RL
abrs_figures_comparison = RB BB BOLA robustMPC RL arte_tree fandom_tree twitch_tree tubitv_tree zdf_tree xvideos_tree srf_tree

video_len = 49

target_testcases = $(foreach trace, $(traces), $(foreach abr, $(abrs), $(testcases_path)/$(abr)-$(trace).case))
target_results = $(foreach trace, $(traces), $(foreach abr, $(abrs), $(results_path)/$(abr)-$(trace)/))

target_figures_validation =  $(results_path)/figs/validation/
target_figures_comparison = $(results_path)/figs/comparison/


target_plots = $(target_figures_validation) $(target_figures_comparison)

getelement = $(word $(2), $(subst -, , $(1)))

targets = $(target_results) $(target_testcases) $(target_plots)


all: ${targets}


help: 
	@echo make track_progress --> how many experiments are done out of the total
	@echo make partial --> computes the partial graphs with the results are already present
	@echo make --> first computes the results, than the final plots
	@echo make clean --> cleans the result directory

track_progress: 
	$(eval files_computed := $(shell ls -l $(results_path) | wc -l))
	$(eval total_files := $(words $(target_results)))
	@echo Experiment status: $(files_computed)/$(total_files) 

$(target_figures_validation): ${target_results} make_plot.py
	python3 make_plot.py 	$(results_path)\
												$(target_figures_validation)\
												$(video_len)\
												--abrs $(abrs_figures_validation)\
												--traces $(traces)\

$(target_figures_comparison): ${target_results} make_plot.py
	python3 make_plot.py 	$(results_path)\
												$(target_figures_comparison)\
												$(video_len)\
												--abrs $(abrs_figures_comparison)\
												--traces $(traces)\

partial : 
	python3 make_plot.py 	$(results_path)\
												$(results_path)/figs/partial/validation\
												$(video_len)\
												--abrs $(abrs_figures_validation)\
												--traces $(traces)\
												--partial\
												--allow_cleaning

	python3 make_plot.py	$(results_path)\
												$(results_path)/figs/partial/comparison\
												$(video_len)\
												--abrs $(abrs_figures_comparison)\
												--traces $(traces)\
												--partial\
												--allow_cleaning

$(results_path)/%/: $(testcases_path)/%.case
	python3 dash_abr_test.py  server_conf/config.json\
														$(testcases_path)/$*.case\
														$(traces_path)\
														$(results_path)/$*\
														--display

$(testcases_path)/%.case: make_testcase.py
	$(eval abr=$(call getelement, $*, 1))
	$(eval trace=$(call getelement, $*, 2))
	
	python3 make_testcase.py 	$(testcases_path)\
														$(trace)\
														videoTest\
														1\
														False\
														$*\
														$(abr)\
														False\
														cubic\
														0\
														1000\
														$(server_path)/$(abr)_server.py

check_clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]


clean: check_clean
	@echo "NOT Removing old experiments"
	@echo "NOT Removing old testcases"




