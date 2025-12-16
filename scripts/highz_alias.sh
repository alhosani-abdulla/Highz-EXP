# Custom aliases for Highz-EXP analysis environment
# Data analysis and plotting using Python 3.11 with scikit-rf

alias GetSpecs='cd /media/peterson/INDURANCE'
alias AnalysisShell='cd /home/peterson/Highz-EXP && pipenv shell'

ViewSpecs() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/digital_spectrometer/viewspec_now.py "$@"
}

export PYTHONPATH=/home/peterson/Highz-EXP/src:$PYTHONPATH

CreatePlot() {
    CURRENT_PATH=$(pwd)
    cd /home/peterson/Highz-EXP
    PLOT_PATH="$CURRENT_PATH/$@"
    pipenv run python src/digital_spectrometer/image_creator.py $PLOT_PATH
}

ViewPlots() {
    cd /home/peterson/Highz-EXP
    pipenv run python src/digital_spectrometer/movie_creator.py "$@"
}
