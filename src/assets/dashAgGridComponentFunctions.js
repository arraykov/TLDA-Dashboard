var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.StockLink = function (props) {
    function openPopup() {
        var url = 'https://www.cefconnect.com/Details/SummaryPrint.aspx?Ticker=' + props.value;
        
        // Modify the width and height values as needed
        var windowFeatures = "width=1000,height=800,scrollbars=yes,resizable=yes,status=yes";
        
        window.open(url, "_blank", windowFeatures);
    }
    
    return React.createElement('a',
    {
        href: '#',
        onClick: openPopup // set the onClick event to our openPopup function
    }, props.value)
}

// Add these functions to your code
dagcomponentfuncs.saveGridState = function(params) {
    localStorage.setItem('columnState', JSON.stringify(params.columnApi.getColumnState()));
}

dagcomponentfuncs.restoreGridState = function(params) {
    var columnState = JSON.parse(localStorage.getItem('columnState'));
    if (columnState) {
        params.columnApi.setColumnState(columnState);
    }
}


window.clickableNavTickerCellRenderer = function(params) {
    var cellElement = document.createElement("a");
    cellElement.href = "#"; // This makes it a hyperlink
    cellElement.innerText = params.value; // Assuming 'params.value' contains the NAV ticker
    cellElement.onclick = function(e) {
        e.preventDefault(); // Prevent the default action (navigation)
        // Update hidden Dash component with the clicked NAV ticker
        var hiddenInput = document.getElementById('hidden-input');
        if (hiddenInput) {
            hiddenInput.value = params.value;
            // Trigger a change event
            var event = new Event('change');
            hiddenInput.dispatchEvent(event);
        }
    };

    return {
        // ag-Grid will call this method to get the DOM element
        getGui: function() {
            return cellElement;
        }
    };
}