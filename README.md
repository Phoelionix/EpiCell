# Epicell - A toy stochastic cellular automaton for modelling epidemics.

(2021 project)

---

This algorithm can simulate multiple scenarios, animate these scenarios and plot multiple variables. It does not in its current state both plot and display an animation, as it is rare for one to need to see both. 

At the start of the program (before the scenarios' variables are set) please modify variables as one wishes. Below is a description of each variable. See the bottom for importantish notes.

Save: Save the animation or plot that is generated.

SaveName: Name for the animation or plot that is generated. Use .gif or .plot for animations and plots respectively.

Frames: The number of days for the simulation to run over.

Runs: The number of runs to perform. When tracking infected against days (usually in the Melbourne scenario) the number of infected on each day is averaged. When tracking the epidemic size or length, for large enough Frames, X runs is almost equivalent to multiplying the Frames by X, as the epidemic sizes/lengths are just added to the list, and the behaviour of the Faroe Islands is SOC.

AnimationSetting: Has multiple meanings
	=False - Plot what is specified in DataPlotType
	=an integer - Plot a number of frames equal to the integer (evenly spaced). If GraphEpidemicOnly = True, the number of frames plotted will be reduced by a factor equal to the percentage of frames during which an epidemic was occurring. For the Faroe Islands, this is about 25%.
	="All" - Plot all frames, or all frames where an epidemic was occurring if GraphEpidemicOnly = True.

DataPlotType: 
	"Rhodes" - Plots a complementary cumulative probability graph of epidemic size (number of people infected) in the same style Rhodes et al. did
	"Infected-Days" - Shows a plot of number of infected against days (averaged over runs).
	"EpidemicLength" Similar to Rhodes, but with epidemic length in days instead of epidemic size. Not included in report but gives a nice scale invariant power law relation!

Scenario:
	"AnimationExample" - A scenario good for seeing the animation in action, 100% susceptible population, multiple regions.
	"Faroe" - Faroe Islands
	"Melbourne" - Melbourne
	"Custom" - For if one wishes to modify the parameters in the custom scenario (found just below this section).

FileLocation: Location of the data for Melbourne's regions. Included in the zip file, no need to change.

LockdownStrategy: Only applicable when Scenario = "Melbourne"
	"MelbourneApproach" - Blanket Lockdown (as discussed in report)
	"SydneyApproach" - Reactive Lockdown (as discussed in report)

ShowIncubating (bool): For the animation, whether incubating individuals are shown (i.e. contribute to redness).

GraphEpidemicOnly (bool): For the animation, whether only frames where individuals are infected or incubating/exposed are shown. Depends on GraphingThreshold.

GraphingThreshold (int): The number of incubating/exposed + infected required for the frame to be animated.

TrackMonthly (bool): Only applicable for when DataPlotType = "Rhodes". Whether or not epidemic size is the actual epidemic size (False) or uses the system Rhodes did - consecutive months with cases are considered a single epidemic (True)

ShowInitialSpace (bool): Whether to print out the arrays for the region in its inital state.

ShowFinalSpace (bool): Whether to print out the arrays for the region in its final state (warning: does this for each run!)

BaseOpacity (0<value<1): The minumum opacity of cells

OpacityRegionsSeparate: Whether the opacity of cells is determined by comparing to within the region (True) or all cells on the map (False)
	Note: I've locked this setting in for the "Melbourne" scenario since it sort of breaks the opacity code if not.


Other notes: 
- There is a tiny chance for an "insignificant error" which is just the map and regions arrays disagreeing on what state one individual is in.
- In Spyder the "Graphing tick ..." message repeats for some reason, due to the animation function. It even happens when the animation function is replaced, so I'm not sure why this happens (I coded in Jupyter), so sorry about that.
- It's recommended to set OpacityRegionsSeparate to True for AnimationExample.
- Animations for Melbourne show little change due to a far lower R value and restrictions.
- I've included some sample outputs, including the AnimationExample animation. I also included "Melbourne Grid Layout.xlsx" to show what each region represents in the animation (the layout has no other effect).
- The simulated Reff is different from the input due to the distributions being approximations. 
- The default immunity for Faroe Islands is 70%. If one wishes to changes this, the variable controlling this is "ImmunePercentage".
