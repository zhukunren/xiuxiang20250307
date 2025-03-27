import streamlit.components.v1 as components
def inject_orientation_script():
    orientation_script = """
    <style>
    #rotate-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        z-index: 9999;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 24px;
    }
    </style>
    <div id="rotate-overlay">
      请旋转手机至横屏模式使用
    </div>
    <script>
    function checkOrientation() {
        if (window.innerHeight > window.innerWidth) {
            document.getElementById('rotate-overlay').style.display = 'flex';
        } else {
            document.getElementById('rotate-overlay').style.display = 'none';
        }
    }
    window.addEventListener('resize', checkOrientation);
    checkOrientation();
    </script>
    """
    components.html(orientation_script, height=0)

def load_custom_css():
    custom_css = """
    <style>
    .strategy-row {
        margin-bottom: 8px;
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    .strategy-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }
    @media only screen and (max-width: 768px) {
        .strategy-row {
            flex-direction: column;
            align-items: flex-start;
        }
        .strategy-label {
            justify-content: flex-start;
            margin-bottom: 4px;
        }
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
            overflow-x: auto;
        }
    }
    </style>
    """