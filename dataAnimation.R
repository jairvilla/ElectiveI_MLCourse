# Source: https://towardsdatascience.com/animating-your-data-visualizations-like-a-boss-using-r-f94ae20843e3


# Install packages ...................
install.packages("plotly")
install.packages("gapminder")
install.packages('htmlwidgets')
# .................................

# Libraries .....................

library(gapminder)
library(plotly)
library(htmlwidgets)
# ................................

names(gapminder)
str(gapminder)


p <- gapminder %>%
  plot_ly(
    x = ~gdpPercap, 
    y = ~lifeExp, 
    size = ~pop, 
    color = ~continent, 
    frame = ~year, 
    text = ~country, 
    hoverinfo = "text",
    type = 'scatter',
    mode = 'markers'
  ) %>%
  layout(
    xaxis = list(
      type = "log"
    )
  )
p
