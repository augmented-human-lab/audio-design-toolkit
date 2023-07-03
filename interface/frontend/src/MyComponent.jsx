import {
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { useEffect, useState } from "react"
import Box from '@material-ui/core/Box'
import { createTheme } from '@material-ui/core/styles';
import { Slider } from "@material-ui/core";
import { ThemeProvider } from '@material-ui/styles';

const debounce = (func, timeout = 1000) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      func.apply(this, args);
    }, timeout);
  };
};

const streamlitSetComponentValue = debounce(newValue => {
  window.dataLayer = window.dataLayer || [];
  function gtag() {
    console.log(arguments)
    window.dataLayer.push(arguments)
  }
  gtag('event', 'parameter_change', {
    'app_name': 'myAppName',
    'screen_name': 'Home',
    'elementId': newValue,
    'value': newValue
  });
  console.log(newValue)
  return Streamlit.setComponentValue(newValue)
}, 500)

const VerticalSlider = (props) => {
  console.log(props)
  const { label, min_value, max_value, value, step, track_color, thumb_color } = props.args;
  // const [min_value, max_value, value, step, track_color, slider_color, thumb_color] = [-5, 5, 0, 0.01, "gray", "red", "black"];
  const theme = props.theme
  // const theme = {
  //   font: 'Serif',
  //   textColor: 'black'
  // }
  const [state, setState] = useState(value)
  useEffect(() => Streamlit.setFrameHeight());
  const handleChange = (_, newValue) => {
    setState(newValue);
    streamlitSetComponentValue(newValue);
  };


  const snowflakeTheme = createTheme({
    overrides: {
      MuiSlider: {
        root: {
          height: 200,
          fontSize: 10,
          marginBottom: 0,
          fontWeight: 400,
          fontFamily: theme.font
        },
        markLabel: {
          color: theme.textColor,
          fontFamily: theme.font,
          paddingLeft: 15,
          fontSize: 14
        },
        markLabelActive: {
          color: theme.textColor,
          fontFamily: theme.font,
          paddingLeft: 15,
          fontSize: 14
        },
        markActive: {
          opacity: 0
        },
        valueLabel: {
          fontFamily: theme.font,
        },
        thumb: {
          color: thumb_color,
          marginLeft: "4px !important"
        },
        track: {
          color: theme.primaryColor,
          width: "10px !important",
          marginLeft: "5px !important",
          borderRadius: 2,
          marginBottom: 0,
          borderWidth: 1
        },
        rail: {
          color: track_color,
          width: "10px !important",
          marginLeft: "5px !important",
          borderRadius: 2,
          marginBottom: 0
        }
      }
    }
  });
  // return <>a</>

  return (
    <Box sx={{ height: 200, marginRight: 10, marginLeft: 10, paddingTop: 10 }}>
      <ThemeProvider theme={snowflakeTheme}>
        <p style={{
          wordBreak: "break-word",
          fontSize: 14,
          marginBottom: '1.5rem',
          fontFamily: theme.font,
          color: theme.textColor
        }}>{label}</p>
        <Slider
          min={min_value}
          step={step}
          max={max_value}
          defaultValue={value}
          value={state}
          onChange={handleChange}
          valueLabelDisplay="on"
          orientation="vertical"
          aria-labelledby="continuous-slider"
          ThumbComponent="span"
          marks={[{ value: Number(min_value), label: String(min_value) }, { value: Number(max_value), label: String(max_value) }]}
        />
      </ThemeProvider>
    </Box>

  );
}

export default
  withStreamlitConnection(VerticalSlider);