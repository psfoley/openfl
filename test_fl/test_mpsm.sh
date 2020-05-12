
make -C ../ build_containers
make -C ../ run_agg_container &
make -C ../ run_col_container &
