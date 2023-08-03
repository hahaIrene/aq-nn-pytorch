
import pyecharts.options as opts
from pyecharts.charts import Line

def generateGraph(epoch, epochs, train_losses, val_losses):
    
    (
        Line()
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
        .add_xaxis(xaxis_data=epochs)
        .add_yaxis(
            series_name="",
            y_axis=train_losses,
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="",
            y_axis=val_losses,
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .render(f"loss-{epoch}.html")
    )
